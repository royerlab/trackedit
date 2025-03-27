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
    time_chunk: int = 0,
    time_chunk_length: int = 105,
    time_chunk_overlap: int = 5,
    allow_overwrite: bool = False,
    imaging_zarr_file: Optional[str] = None,
    imaging_channel: Optional[str] = None,
    viewer: Optional[napari.Viewer] = None,
) -> Tuple[napari.Viewer, TrackEditClass]:
    """Run TrackEdit on a database file.

    Args:
        working_directory: Path to working directory
        db_filename: Name of database file
        tmax: Maximum time point
        scale: Scale factors for each dimension
        name: Name for the tracks layer
        time_chunk: Starting time chunk
        time_chunk_length: Length of time chunks
        time_chunk_overlap: Overlap between time chunks
        allow_overwrite: Allow overwriting existing files
        imaging_zarr_file: Path to imaging zarr file
        imaging_channel: Channel to use from imaging file
        viewer: Optional existing napari viewer

    Returns:
        Tuple[napari.Viewer, TrackEditClass]: The viewer instance and TrackEdit instance
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
        time_chunk=time_chunk,
        time_chunk_length=time_chunk_length,
        time_chunk_overlap=time_chunk_overlap,
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

    track_edit = TrackEditClass(viewer, databasehandler=DB_handler)
    viewer.dims.ndisplay = 3  # 3D view
    wrap_default_widgets_in_tabs(viewer)
    viewer.dims.current_step = (2, *viewer.dims.current_step[1:])

    # Set grid parameters without activating grid view
    viewer.grid.shape = (1, -1)
    viewer.grid.stride = 3

    if not viewer_provided:
        napari.run()

    return track_edit
