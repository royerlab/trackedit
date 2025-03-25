from pathlib import Path
from typing import Optional, Tuple

import napari

from motile_tracker.data_model.actions import (
    AddEdges,
    AddNodes,
    DeleteEdges,
    DeleteNodes,
)
from motile_tracker.data_views import TracksViewer
from trackedit.DatabaseHandler import DatabaseHandler
from trackedit.motile_overwrites import (
    create_db_add_edges,
    create_db_add_nodes,
    create_db_delete_edges,
    create_db_delete_nodes,
    create_tracks_viewer_and_segments_refresh,
)
from trackedit.TrackEditClass import TrackEditClass
from trackedit.utils.utils import wrap_default_widgets_in_tabs


def run_trackedit(
    working_directory: Path,
    db_filename: str = "data.db",
    tmax: int = None,
    scale: Tuple[float, ...] = (1, 1, 1),
    layer_name: str = "ultrack",
    allow_overwrite: bool = False,
    imaging_zarr_file: Optional[str] = None,
    imaging_channel: Optional[str] = None,
    viewer: Optional[napari.Viewer] = None,
) -> napari.Viewer:
    """
    Run TrackEdit with the specified parameters.

    Parameters
    ----------
    working_directory : Path
        Path to the working directory containing the database file and metadata.toml
    db_filename : str, optional
        Name of the database file to start from, by default "data.db"
    tmax : int, optional
        Maximum number of frames to display, by default None (use all frames)
    scale : Tuple[float, ...], optional
        Scale factors for (Z),Y,X dimensions, by default (1, 1, 1)
    layer_name : str, optional
        Name of the layer in napari, by default "ultrack"
    allow_overwrite : bool, optional
        Whether to overwrite existing database/changelog, by default False
    imaging_zarr_file : str, optional
        Path to zarr file containing imaging data, by default None
    imaging_channel : str, optional
        Channel specification for imaging data, by default None
    viewer : Optional[napari.Viewer], optional
        Existing napari viewer to use, by default None
    """
    viewer_provided = viewer is not None
    if not viewer_provided:
        viewer = napari.Viewer()

    DB_handler = DatabaseHandler(
        db_filename_old=db_filename,
        working_directory=working_directory,
        Tmax=tmax,
        scale=scale,
        name="ultrack",
        allow_overwrite=allow_overwrite,
        imaging_zarr_file=imaging_zarr_file,
        imaging_channel=imaging_channel,
    )

    # overwrite some motile functions
    DeleteNodes._apply = create_db_delete_nodes(DB_handler)
    DeleteEdges._apply = create_db_delete_edges(DB_handler)
    AddEdges._apply = create_db_add_edges(DB_handler)
    AddNodes._apply = create_db_add_nodes(DB_handler)
    TracksViewer._refresh = create_tracks_viewer_and_segments_refresh(
        layer_name=layer_name
    )

    _trackeditclass = TrackEditClass(viewer, databasehandler=DB_handler)
    viewer.dims.ndisplay = 3  # 3D view
    wrap_default_widgets_in_tabs(viewer)
    viewer.dims.current_step = (2, *viewer.dims.current_step[1:])

    if not viewer_provided:
        napari.run()

    return viewer
