import multiprocessing as mp
import warnings
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import napari
import numpy as np
import pandas as pd
import sqlalchemy as sqla
import zarr
from qtpy.QtWidgets import QWidget
from sqlalchemy.orm import Session
from toolz import curry
from tqdm import tqdm
from ultrack.config.config import MainConfig
from ultrack.config.dataconfig import DataConfig
from ultrack.core.database import NodeDB
from ultrack.utils.array import create_zarr, large_chunk_size
from zarr.storage import Store


def wrap_default_widgets_in_tabs(viewer: napari.Viewer):
    """Wrap the default napari widgets in tabs"""
    # Use proper API to access layers list container
    list_container = viewer.window._qt_window.findChild(QWidget, "LayerList")
    if list_container:
        viewer.window._qt_window.remove_dock_widget(list_container)
        viewer.window.add_dock_widget(
            list_container, name="layers", area="right", allowed_areas=["right"]
        )


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
