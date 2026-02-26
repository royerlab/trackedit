import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import sqlalchemy as sqla
import zarr
from qtpy.QtWidgets import QTabWidget
from sqlalchemy.orm import Session
from toolz import curry
from tqdm import tqdm
from ultrack.config.config import MainConfig
from ultrack.config.dataconfig import DataConfig
from ultrack.core.database import NodeDB
from ultrack.utils.array import create_zarr, large_chunk_size
from ultrack.utils.constants import NO_PARENT
from zarr.storage import StoreLike


def wrap_default_widgets_in_tabs(viewer):
    # -- 1) Identify the default dock widgets by going up the parent chain.
    # For controls: the dock widget is the direct parent.
    controls_dock = viewer.window.qt_viewer.controls.parentWidget()
    # For the layer list: go up two levels.
    list_dock = viewer.window.qt_viewer.layers.parentWidget().parentWidget()

    # -- 2) Instead of only taking the inner widget,
    # retrieve the entire container from the dock widget.
    controls_container = controls_dock.widget() if controls_dock else None
    list_container = list_dock.widget() if list_dock else None

    # -- 3) Remove the dock widgets from the main window,
    # but do not close or delete them so that Napari's internal references remain valid.
    main_window = viewer.window._qt_window
    for dock in [controls_dock, list_dock]:
        if dock is not None:
            main_window.removeDockWidget(dock)

    # -- 4) Detach the container widgets from their docks
    if controls_container:
        controls_container.setParent(None)
    if list_container:
        list_container.setParent(None)

    # -- 5) Create a tab widget and add the containers as tabs.
    tab_widget = QTabWidget()
    if controls_container:
        tab_widget.addTab(controls_container, "Layer Controls")
    if list_container:
        tab_widget.addTab(list_container, "Layer List")

    # -- 6) Add our new tab widget as a dock widget.
    new_dock = viewer.window.add_dock_widget(tab_widget, area="left", name="napari")

    # -- 7) (Optional) Update internal viewer references so that
    # Napari's menu actions refer to the new widgets.
    viewer.window.qt_viewer._controls = controls_container
    viewer.window.qt_viewer._layers = list_container

    # (Optional) Also update the internal dict of dock widgets if needed:
    viewer.window._dock_widgets["Layer List"] = new_dock


@curry
def _query_and_export_data_to_frame(
    time: int,
    database_path: str,
    shape: Tuple[int],
    df: pd.DataFrame,
    export_func: Callable[[int, np.ndarray], None],
) -> None:
    """Queries segmentation data from database and paints it according to their respective `df` `track_id` column.

    Parameters
    ----------
    time : int
        Frame time point to paint.
    database_path : str
        Database path.
    shape : Tuple[int]
        Frame shape.
    df : pd.DataFrame
        Tracks dataframe.
    export_func : Callable[[int, np.ndarray], None]
        Export function, it receives as input a time index `t` and its respective uint16 labeled buffer.
    """
    node_indices = set(df[df["t"] == time].index)

    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        buffer = np.zeros(shape, dtype=int)
        query = list(
            session.query(NodeDB.id, NodeDB.pickle).where(
                NodeDB.t == time, NodeDB.selected
            )
        )

        if len(query) == 0:
            warnings.warn(f"Segmentation mask from t = {time} is empty.")

        for id, node in query:
            if id not in node_indices:
                # ignoring nodes not present in dataset, used when exporting a subset of data
                # filtering through a sql query crashed with big datasets
                continue

            generic = df.loc[id, "generic"]
            node.paint_buffer(buffer, value=generic, include_time=False)

        export_func(time, buffer)


def multiprocessing_apply(
    func: Callable[[Any], None],
    sequence: Sequence[Any],
    n_workers: int,
    desc: Optional[str] = None,
) -> List[Any]:
    """Applies `func` for each item in `sequence`.

    Parameters
    ----------
    func : Callable[[Any], NoneType]
        Function to be executed.
    sequence : Sequence[Any]
        Sequence of parameters.
    n_workers : int
        Number of workers for multiprocessing.
    desc : Optional[str], optional
        Description to tqdm progress bar, by default None

    Returns
    -------
    List[int]
        List of `func` outputs.
    """
    length = len(sequence)
    if n_workers > 1 and length > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(n_workers, length)) as pool:
            return list(tqdm(pool.imap(func, sequence), desc=desc, total=length))
    else:
        return [func(t) for t in tqdm(sequence, desc=desc)]


def export_annotation_generic(
    data_config: DataConfig,
    df: pd.DataFrame,
    export_func: Callable[[int, np.ndarray], None],
) -> None:
    """
    Generic function to export segmentation masks, segments labeled by `track_id` from `df`.

    Parameters
    ----------
    data_config : DataConfig
        Data parameters configuration.
    df : pd.DataFrame
        Tracks dataframe indexed by node id.
    export_func : Callable[[int, np.ndarray], None]
        Export function, it receives as input a time index `t` and its respective uint16 labeled buffer.
    """

    if "track_id" not in df.columns:
        raise ValueError(f"Dataframe must have `track_id` column. Found {df.columns}")

    shape = data_config.metadata["shape"]

    multiprocessing_apply(
        _query_and_export_data_to_frame(
            database_path=data_config.database_path,
            shape=shape[1:],
            df=df,
            export_func=export_func,
        ),
        sequence=range(shape[0]),
        n_workers=data_config.n_workers,
        desc="Exporting annotations",
    )


def annotations_to_zarr(
    config: MainConfig,
    tracks_df: pd.DataFrame,
    store_or_path: Union[None, StoreLike, Path, str] = None,
    chunks: Optional[Tuple[int]] = None,
    overwrite: bool = False,
) -> zarr.Array:
    """
    Exports segmentations masks to zarr array, `track_df` assign the `track_id` to their respective segments.
    By changing the `store` this function can be used to write zarr arrays into disk.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    tracks_df : pd.DataFrame
        Tracks dataframe, must have `track_id` column and be indexed by node id.
    store_or_path : Union[None, StoreLike, Path, str], optional
        Zarr storage or output path, if not provided zarr.TempStore is used.
    chunks : Optional[Tuple[int]], optional
        Chunk size, if not provided it chunks time with 1 and the spatial dimensions as big as possible.
    overwrite : bool, optional
        If True, overwrites existing zarr array.

    Returns
    -------
    zarr.Array
        Output zarr array.
    """

    shape = config.data_config.metadata["shape"]
    dtype = np.int32

    if (
        isinstance(store_or_path, zarr.storage.MemoryStore)
        and config.data_config.n_workers > 1
    ):
        raise ValueError(
            "zarr.MemoryStore and multiple workers are not allowed. "
            f"Found {config.data_config.n_workers} workers in `data_config`."
        )

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=dtype)

    if isinstance(store_or_path, StoreLike):
        array = zarr.zeros(shape, dtype=dtype, store=store_or_path, chunks=chunks)
    else:
        array = create_zarr(
            shape,
            dtype=dtype,
            store_or_path=store_or_path,
            chunks=chunks,
            default_store_type=zarr.TempStore,
        )

    export_annotation_generic(config.data_config, tracks_df, array.__setitem__)
    return array


def solution_dataframe_from_sql_with_tmax(
    database_path: str,
    tmax: int,
    columns: Sequence[sqla.Column] = (
        NodeDB.id,
        NodeDB.parent_id,
        NodeDB.t,
        NodeDB.z,
        NodeDB.y,
        NodeDB.x,
    ),
) -> pd.DataFrame:
    """Query `columns` of nodes in current solution (NodeDB.selected == True).

    Parameters
    ----------
    database_path : str
        SQL database path (e.g. sqlite:///your.database.db)

    columns : Sequence[sqla.Column], optional
        Queried columns, MUST include NodeDB.id.
        By default (NodeDB.id, NodeDB.parent_id, NodeDB.t, NodeDB.z, NodeDB.y, NodeDB.x)

    Returns
    -------
    pd.DataFrame
        Solution dataframe indexed by NodeDB.id
    """

    tmax = int(tmax)

    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        statement = (
            session.query(*columns)
            .where(NodeDB.selected)
            .where(NodeDB.t < tmax)
            .statement
        )
        df = pd.read_sql(statement, session.bind, index_col="id")

    return df


def remove_nonexisting_parents(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove parents that do not exist in the dataframe.
    Why:
    Let's say we have a database with 200 frames, with tracks spanning all 200 frames.
    But if the database is opened with tMax=100, and a cell i is deleted at t=100,
    the parent_id of its parent cell (at t=101) in the database is still i.
    If later, the database is opened with tMax=200, the parent_id of the cell at t=101 is still i,
    but cell i no longer exist, giving an error when assigning track_ids.

    This function set the parent_ids to -1, for the rows in the database whose
    parent_id is not in the dataframe (selected=False)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with columns: id, parent_id

    Returns
    -------
    pd.DataFrame
        Dataframe with the parent_ids set to -1 for the rows in the database whose
        parent_id is not in the dataframe (selected=False)
    """
    all_parent_ids = df["parent_id"].to_numpy()
    all_ids = df.index.to_numpy()
    non_existing_parent_ids = np.setdiff1d(all_parent_ids, all_ids)
    non_existing_parent_ids = non_existing_parent_ids[
        non_existing_parent_ids != NO_PARENT
    ]
    df.loc[df.parent_id.isin(non_existing_parent_ids), "parent_id"] = NO_PARENT
    return df


def create_ellipsoid_mask(center, radii, shape):
    """Create an ellipsoid mask at given center with given radii.

    This is useful for creating anisotropic shapes that compensate for
    anisotropic voxel spacing/scaling.

    Parameters
    ----------
    center : tuple
        Center coordinates (z, y, x) for 3D or (y, x) for 2D
    radii : tuple
        Radii in each dimension (z, y, x) for 3D or (y, x) for 2D
    shape : tuple
        Shape of output array (spatial dimensions only)

    Returns
    -------
    mask : ndarray (bool)
        Boolean mask of the ellipsoid
    """
    assert len(center) == len(
        shape
    ), f"Center {center} and shape {shape} dimensions must match"
    assert len(radii) == len(
        shape
    ), f"Radii {radii} and shape {shape} dimensions must match"

    # Create index grid
    indices = np.moveaxis(np.indices(shape), 0, -1)

    # Calculate normalized distances from center
    # For each dimension, divide by the radius for that dimension
    center_arr = np.asarray(center)
    radii_arr = np.asarray(radii)

    # Compute (index - center) / radius for each dimension
    normalized_distances = (indices - center_arr) / radii_arr

    # Sum of squared normalized distances
    distance_squared = np.sum(normalized_distances**2, axis=-1)

    # Create boolean mask (inside ellipsoid if distance^2 <= 1)
    mask = distance_squared <= 1.0

    return mask.astype(bool)


def calculate_bbox_from_mask(mask):
    """Calculate bounding box from boolean mask.

    Parameters
    ----------
    mask : ndarray (bool)
        Boolean mask

    Returns
    -------
    bbox : ndarray (int32)
        Bounding box in format [min_d0, min_d1, ..., max_d0, max_d1, ...]
    """
    coords = np.argwhere(mask)

    if len(coords) == 0:
        raise ValueError("Mask is empty, cannot calculate bounding box")

    # Get min and max for each dimension
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    # Concatenate into bbox format
    bbox = np.concatenate([mins, maxs]).astype(np.int32)

    return bbox


def create_cell_mask_and_bbox(
    position_scaled: np.ndarray,
    radius_pixels: float,
    ndim: int,
    scale: tuple,
    data_shape_full: tuple,
) -> tuple:
    """Create mask and bbox for a cell at the given position.

    Parameters
    ----------
    position_scaled : array-like
        Position in viewer coordinates (scaled)
    radius_pixels : float
        Radius of the sphere in pixels
    ndim : int
        Number of dimensions (3 for 2D+t, 4 for 3D+t)
    scale : tuple
        Scale factors (z_scale, y_scale, x_scale) for 4D or (y_scale, x_scale) for 3D
    data_shape_full : tuple
        Full data shape including time dimension

    Returns
    -------
    tuple of (mask, bbox) or (None, None) if creation failed
    """
    from napari.utils.notifications import show_warning

    position_scaled = np.array(position_scaled)

    # Unscale coordinates from viewer to database space
    if ndim == 4:
        z_scale, y_scale, x_scale = scale
        center_db = np.array(
            [
                position_scaled[0] / z_scale,
                position_scaled[1] / y_scale,
                position_scaled[2] / x_scale,
            ]
        )
        radii = (
            radius_pixels / z_scale,
            radius_pixels / y_scale,
            radius_pixels / x_scale,
        )
    else:
        y_scale, x_scale = scale
        center_db = np.array(
            [
                0,  # z placeholder
                position_scaled[0] / y_scale,
                position_scaled[1] / x_scale,
            ]
        )
        radii = (
            radius_pixels / y_scale,
            radius_pixels / x_scale,
        )

    # Calculate bounding box
    shape = data_shape_full[1:]
    if ndim == 4:
        radius_z = int(np.ceil(radii[0]))
        radius_y = int(np.ceil(radii[1]))
        radius_x = int(np.ceil(radii[2]))
        bbox = np.array(
            [
                max(0, int(center_db[0]) - radius_z),
                max(0, int(center_db[1]) - radius_y),
                max(0, int(center_db[2]) - radius_x),
                min(shape[0], int(center_db[0]) + radius_z + 1),
                min(shape[1], int(center_db[1]) + radius_y + 1),
                min(shape[2], int(center_db[2]) + radius_x + 1),
            ],
            dtype=np.int32,
        )
        mask_shape = (bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2])
        mask_center = (
            center_db[0] - bbox[0],
            center_db[1] - bbox[1],
            center_db[2] - bbox[2],
        )
    else:
        radius_y = int(np.ceil(radii[0]))
        radius_x = int(np.ceil(radii[1]))
        bbox = np.array(
            [
                max(0, int(center_db[1]) - radius_y),
                max(0, int(center_db[2]) - radius_x),
                min(shape[1], int(center_db[1]) + radius_y + 1),
                min(shape[2], int(center_db[2]) + radius_x + 1),
            ],
            dtype=np.int32,
        )
        mask_shape = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        mask_center = (center_db[1] - bbox[0], center_db[2] - bbox[1])

    # Create mask
    try:
        mask = create_ellipsoid_mask(center=mask_center, radii=radii, shape=mask_shape)
        if not mask.any():
            show_warning("Created mask is empty - position may be out of bounds")
            return None, None
        return mask, bbox
    except (AssertionError, ValueError) as e:
        show_warning(f"Failed to create mask: {e}")
        return None, None


def fix_overlap_ancestor_ids(database_path: str, new_node_id: int, current_time: int):
    """Fix overlap entries with ancestor_id=-1 due to ultrack pickle bug.

    Parameters
    ----------
    database_path : str
        Path to the database
    new_node_id : int
        ID of the newly added node
    current_time : int
        Current time frame

    Notes
    -----
    TODO: This is no longer necessary once Ultrack fixes the ancestor_id bug
    """
    from ultrack.core.database import NodeDB, OverlapDB

    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        overlaps_to_fix = (
            session.query(OverlapDB)
            .filter(OverlapDB.node_id == new_node_id, OverlapDB.ancestor_id == -1)
            .all()
        )

        if not overlaps_to_fix:
            return

        new_node_row = session.query(NodeDB).filter(NodeDB.id == new_node_id).first()
        new_node = new_node_row.pickle

        neighbor_rows = (
            session.query(NodeDB)
            .filter(NodeDB.t == current_time, NodeDB.id != new_node_id)
            .all()
        )

        actual_overlaps = []
        for neighbor_row in neighbor_rows:
            iou = new_node.IoU(neighbor_row.pickle)
            if iou > 0.0:
                actual_overlaps.append((neighbor_row.id, iou))

        actual_overlaps.sort(key=lambda x: x[1], reverse=True)

        for i, overlap in enumerate(overlaps_to_fix):
            if i < len(actual_overlaps):
                overlap.ancestor_id = actual_overlaps[i][0]
            else:
                session.delete(overlap)

        session.commit()
