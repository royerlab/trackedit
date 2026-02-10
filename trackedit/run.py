from pathlib import Path
from typing import List, Optional, Tuple

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
    work_in_existing_db: bool = False,
    imaging_zarr_file: Optional[str] = None,
    imaging_channel: Optional[str] = None,
    image_z_slice: Optional[int] = None,
    image_translate: Optional[Tuple[float, ...]] = None,
    viewer: Optional[napari.Viewer] = None,
    flag_show_hierarchy: bool = True,
    flag_allow_adding_spherical_cell: bool = False,
    flag_remove_red_flags_at_edge: bool = False,
    remove_red_flags_at_edge_threshold: int = 10,
    annotation_mapping: Optional[dict] = None,
    coordinate_filters: Optional[list] = None,
    default_start_annotation: Optional[int] = None,
    imaging_layer_names: Optional[List[str]] = None,
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
        work_in_existing_db: Work in existing database (only used if db_filename is specifically specified)
        allow_overwrite: Allow overwriting existing files
        imaging_zarr_file: Path to imaging zarr file
        imaging_channel: Channel to use from imaging file
        viewer: Optional existing napari viewer
        flag_show_hierarchy: Show hierarchy in the viewer
        flag_allow_adding_spherical_cell: Allow adding spherical cells via button (default: False)
        annotation_mapping: Mapping of annotation ids to names and colors
        imaging_layer_names: Names for imaging layers. If None, defaults to ['nuclear', 'membrane'] for 2 channels

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
        annotation_mapping=annotation_mapping,
        time_chunk=time_chunk,
        time_chunk_length=time_chunk_length,
        time_chunk_overlap=time_chunk_overlap,
        allow_overwrite=allow_overwrite,
        work_in_existing_db=work_in_existing_db,
        imaging_zarr_file=imaging_zarr_file,
        imaging_channel=imaging_channel,
        image_z_slice=image_z_slice,
        image_translate=image_translate,
        coordinate_filters=coordinate_filters,
        default_start_annotation=default_start_annotation,
        imaging_layer_names=imaging_layer_names,
        flag_remove_red_flags_at_edge=flag_remove_red_flags_at_edge,
        remove_red_flags_at_edge_threshold=remove_red_flags_at_edge_threshold,
    )

    # overwrite some motile functions
    DeleteNodes._apply = create_db_delete_nodes(DB_handler)
    DeleteEdges._apply = create_db_delete_edges(DB_handler)
    AddEdges._apply = create_db_add_edges(DB_handler)
    AddNodes._apply = create_db_add_nodes(DB_handler)
    TracksViewer._refresh = create_tracks_viewer_and_segments_refresh(
        layer_name=layer_name
    )

    track_edit = TrackEditClass(
        viewer,
        databasehandler=DB_handler,
        flag_show_hierarchy=flag_show_hierarchy,
        flag_allow_adding_spherical_cell=flag_allow_adding_spherical_cell,
    )
    if DB_handler.ndim == 4:
        viewer.dims.ndisplay = 3  # 3D view
    wrap_default_widgets_in_tabs(viewer)
    viewer.dims.current_step = (2, *viewer.dims.current_step[1:])

    # Set grid parameters without activating grid view
    viewer.grid.shape = (1, -1)
    viewer.grid.stride = 3

    if not viewer_provided:
        napari.run()

    return track_edit
