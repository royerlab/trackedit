from pathlib import Path

import napari
import pytest
from qtpy.QtWidgets import QTabWidget, QWidget

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

    run_trackedit(
        working_directory=working_directory,
        db_filename="data.db",
        tmax=5,
        scale=(1, 1, 1),
        allow_overwrite=True,
        imaging_zarr_file="",
        imaging_channel="",
        viewer=viewer,
    )

    return viewer, qtbot


@pytest.mark.parametrize(
    "timelapse_mock_data",
    [
        {"length": 5, "size": 64, "n_dim": 3},
    ],
    indirect=True,
)
def test_ui_interactions(viewer_and_trackedit):
    """Test UI interactions with the viewer and widgets"""
    viewer, qtbot = viewer_and_trackedit

    # Get handles to important widgets
    trackedit_dock = viewer.window._qt_window.findChild(QWidget, "TrackEdit Widgets")
    assert trackedit_dock is not None, "TrackEdit dock widget not found"

    # Find the tab widget and get the Navigation tab
    tab_widget = trackedit_dock.findChild(QTabWidget)
    assert tab_widget is not None, "Tab widget not found"

    # Get the Navigation widget from the first tab
    navigation_widget = tab_widget.widget(0)  # First tab (index 0) is Navigation
    assert navigation_widget is not None, "NavigationWidget not found"

    time_box = trackedit_dock.findChild(QWidget, "TimeBox")
    assert time_box is not None, "TimeBox not found"

    red_flag_box = trackedit_dock.findChild(QWidget, "RedFlagBox")
    assert red_flag_box is not None, "RedFlagBox not found"

    division_box = trackedit_dock.findChild(QWidget, "DivisionBox")
    assert division_box is not None, "DivisionBox not found"

    TV = navigation_widget.tracks_viewer

    # # Test: Select cell
    num_selected_before = len(TV.selected_nodes)
    TV.selected_nodes.add(3000009, append=False)
    num_selected_after = len(TV.selected_nodes)
    TV.selected_nodes.reset()
    print("num_selected_cells", num_selected_before, num_selected_after)
    assert num_selected_after == num_selected_before + 1, "Cell selection failed"

    # # Test: Select multiple cells
    num_selected_before = len(TV.selected_nodes)
    TV.selected_nodes.add(3000006, append=True)
    TV.selected_nodes.add(3000009, append=True)
    num_selected_after = len(TV.selected_nodes)
    TV.selected_nodes.reset()
    print("num_selected_cells", num_selected_before, num_selected_after)
    assert num_selected_after == num_selected_before + 2, "Cell deselection failed"

    # # Test: delete cell
    TV.selected_nodes.add(3000009, append=False)
    TV.delete_node()
    TV.undo()

    # # Test undo/redo
    TV.redo()
    TV.undo()

    # # Test: break/add edge
    TV.selected_nodes.add(2000009, append=False)
    TV.selected_nodes.add(3000010, append=False)
    TV.create_edge()

    # Keep the viewer open
    viewer.window.show()
    qtbot.stop()
