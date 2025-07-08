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
from zarr.storage import Store


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
    store_or_path: Union[None, Store, Path, str] = None,
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
    store_or_path : Union[None, Store, Path, str], optional
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

    if isinstance(store_or_path, zarr.MemoryStore) and config.data_config.n_workers > 1:
        raise ValueError(
            "zarr.MemoryStore and multiple workers are not allowed. "
            f"Found {config.data_config.n_workers} workers in `data_config`."
        )

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=dtype)

    if isinstance(store_or_path, Store):
        array = zarr.zeros(shape, dtype=dtype, store=store_or_path, chunks=chunks)

    else:
        array = create_zarr(
            shape,
            dtype=dtype,
            store_or_path=store_or_path,
            chunks=chunks,
            default_store_type=zarr.TempStore,
            overwrite=overwrite,
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
