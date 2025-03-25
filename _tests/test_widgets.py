from pathlib import Path

import napari
import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QPushButton

# from qtpy.QtGui import QPoint
from ultrack.config import MainConfig
from ultrack.utils.test_utils import *  # Import all ultrack test utilities/fixtures

from trackedit.run import run_trackedit


@pytest.fixture
def viewer_and_trackedit(tracked_database_mock_data: MainConfig, qtbot):
    """Fixture to create viewer and trackedit instance for testing"""
    viewer = napari.Viewer()
    data_config = tracked_database_mock_data.data_config
    working_directory = Path(data_config.working_dir)

    track_edit = run_trackedit(
        working_directory=working_directory,
        db_filename="data.db",
        tmax=20,
        scale=(1, 1, 1),
        allow_overwrite=True,
        time_chunk_length=15,
        imaging_zarr_file="",
        imaging_channel="",
        viewer=viewer,
    )

    return viewer, track_edit, qtbot


@pytest.mark.parametrize(
    "timelapse_mock_data",
    [
        {"length": 20, "size": 64, "n_dim": 3},
    ],
    indirect=True,
)
def test_ui_interactions(viewer_and_trackedit):
    """Test UI interactions with the viewer and widgets"""
    viewer, track_edit, qtbot = viewer_and_trackedit

    # Get the NavigationWidget directly from TrackEdit instance
    navigation_widget = track_edit.NavigationWidget
    assert navigation_widget is not None, "NavigationWidget not found"

    # Get boxes through the properly connected NavigationWidget
    time_box = navigation_widget.time_box
    assert time_box is not None, "TimeBox not found"

    red_flag_box = navigation_widget.red_flag_box
    assert red_flag_box is not None, "RedFlagBox not found"

    division_box = navigation_widget.division_box
    assert division_box is not None, "DivisionBox not found"

    TV = track_edit.tracksviewer

    check_selection(TV)
    check_time_box(time_box)
    check_editing(TV)
    check_red_flag_box(red_flag_box)

    # Keep the viewer open
    viewer.window.show()
    qtbot.stop()


def check_selection(TV):
    """Check cell selection functionality"""

    # Test: Select single cell
    num_selected_before = len(TV.selected_nodes)
    TV.selected_nodes.add(3000009, append=False)
    num_selected_after = len(TV.selected_nodes)
    TV.selected_nodes.reset()
    print("num_selected_cells", num_selected_before, num_selected_after)
    assert num_selected_after == num_selected_before + 1, "Cell selection failed"

    # Test: Select multiple cells
    num_selected_before = len(TV.selected_nodes)
    TV.selected_nodes.add(3000006, append=True)
    TV.selected_nodes.add(3000009, append=True)
    num_selected_after = len(TV.selected_nodes)
    TV.selected_nodes.reset()
    print("num_selected_cells", num_selected_before, num_selected_after)
    assert (
        num_selected_after == num_selected_before + 2
    ), "Multiple cell selection failed"


def check_time_box(time_box):
    """Check time box functionality"""

    time_box.time_input.setText(str(19))
    time_box.on_time_input_entered()

    time_box.time_input.setText(str(2))
    time_box.on_time_input_entered()


def check_editing(TV):
    """Check track editing functionality"""
    # Test: delete cell and undo
    TV.selected_nodes.add(3000009, append=False)
    TV.delete_node()
    TV.undo()

    # Test: redo and undo deletion
    TV.redo()
    TV.undo()

    # Function to handle dialog
    def handle_dialog():
        for widget in QApplication.topLevelWidgets():
            if widget.windowTitle() == "Delete existing edge?":
                for button in widget.findChildren(QPushButton):
                    if button.text() == "&OK":
                        button.click()
                        break

    # Test: break/add edge
    TV.selected_nodes.add(2000009, append=False)
    TV.selected_nodes.add(3000010, append=True)
    QTimer.singleShot(100, handle_dialog)  # handle popup window
    TV.create_edge()


def check_red_flag_box(red_flag_box):
    """Check red flag box functionality"""
    # red_flag_box.red_flag_button.click()
    # red_flag_box.red_flag_button.click()
