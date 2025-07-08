from pathlib import Path
from typing import Callable

import napari
import pytest
from ultrack.config import MainConfig
from ultrack.utils.test_utils import (  # noqa: F401, F811
    config_content,
    config_instance,
    config_path,
    linked_database_mock_data,
    segmentation_database_mock_data,
    timelapse_mock_data,
    tracked_database_mock_data,
)

from trackedit.run import run_trackedit


@pytest.mark.parametrize(
    "timelapse_mock_data,scale,tmax",  # noqa: F811
    [
        ({"length": 4, "size": 64, "n_dim": 3}, (1, 1, 1), 5),
    ],
    indirect=["timelapse_mock_data"],
)
def test_trackedit_widgets(
    tracked_database_mock_data: MainConfig,  # noqa: F811
    timelapse_mock_data,  # noqa: F811
    make_napari_viewer: Callable[[], napari.Viewer],
    request,
    scale,
    tmax,
):
    """Test UI interactions with the viewer and widgets"""

    # Create a new viewer for each test
    viewer = make_napari_viewer()
    data_config = tracked_database_mock_data.data_config
    working_directory = Path(data_config.working_dir)

    track_edit = run_trackedit(
        working_directory=working_directory,
        db_filename="data.db",
        tmax=2,
        scale=scale,
        allow_overwrite=True,
        time_chunk_length=3,
        time_chunk_overlap=1,
        imaging_zarr_file="",
        imaging_channel="",
        viewer=viewer,
    )

    # Get the NavigationWidget directly from TrackEdit instance
    navigation_widget = track_edit.NavigationWidget
    assert navigation_widget is not None, "NavigationWidget not found"

    editing_menu = track_edit.EditingMenu
    assert editing_menu is not None, "EditingMenu not found"

    # Get boxes through the properly connected NavigationWidget
    time_box = navigation_widget.time_box
    assert time_box is not None, "TimeBox not found"

    red_flag_box = navigation_widget.red_flag_box
    assert red_flag_box is not None, "RedFlagBox not found"

    division_box = navigation_widget.division_box
    assert division_box is not None, "DivisionBox not found"

    toAnnotateBox = track_edit.AnnotationWidget.toannotate_box
    assert toAnnotateBox is not None, "ToAnnotateBox not found"

    if request.config.getoption("--show-napari-viewer"):
        napari.run()
