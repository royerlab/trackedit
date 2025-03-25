from pathlib import Path

import napari
import pytest
from qtpy.QtWidgets import QWidget
from ultrack.config import MainConfig
from ultrack.utils.test_utils import *  # Import all ultrack test utilities/fixtures

from trackedit.run import run_trackedit


@pytest.fixture
def viewer_and_trackedit(tracked_database_mock_data: MainConfig, qtbot):
    """Fixture to create viewer and trackedit instance for testing"""
    viewer = napari.Viewer()
    data_config = tracked_database_mock_data.data_config
    working_directory = Path(data_config.working_dir)

    run_trackedit(
        working_directory=working_directory,
        db_filename="data.db",
        tmax=3,
        scale=(1, 1, 1),
        allow_overwrite=True,
        imaging_zarr_file="",
        imaging_channel="",
        viewer=viewer,
    )

    return viewer, qtbot


def test_ui_interactions(viewer_and_trackedit):
    """Test UI interactions with the viewer and widgets"""
    viewer, qtbot = viewer_and_trackedit

    # Get handles to important widgets using proper API
    trackedit_dock = viewer.window._qt_window.findChild(QWidget, "TrackEdit Widgets")
    assert trackedit_dock is not None, "TrackEdit dock widget not found"

    # Find widgets by their object names
    red_flag_box = trackedit_dock.findChild(QWidget, "RedFlagBox")
    assert red_flag_box is not None, "RedFlagBox not found"

    division_box = trackedit_dock.findChild(QWidget, "DivisionBox")
    assert division_box is not None, "DivisionBox not found"

    # Keep the viewer open
    viewer.window.show()
    qtbot.stop()


# def test_opening(
#     tracked_database_mock_data: MainConfig,
# ) -> None:
#     """Original test kept for reference"""
#     print("opening")
#     data_config = tracked_database_mock_data.data_config
#     working_directory = Path(data_config.working_dir)

#     run_trackedit(
#         working_directory=working_directory,
#         db_filename="data.db",
#         tmax=3,
#         scale=(1, 1, 1),
#         allow_overwrite=True,
#         imaging_zarr_file="",
#         imaging_channel="",
#     )

#     assert 1==1
