from pathlib import Path
from typing import Callable

import napari
import pytest
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QApplication, QPushButton
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


@pytest.fixture
def viewer_and_trackedit(
    tracked_database_mock_data: MainConfig,  # noqa: F811
    make_napari_viewer: Callable[[], napari.Viewer],
):
    """Fixture to create viewer and trackedit instance for testing"""
    viewer = make_napari_viewer()
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

    return viewer, track_edit


@pytest.mark.parametrize(
    "timelapse_mock_data",  # noqa: F811
    [
        {"length": 20, "size": 64, "n_dim": 3},
    ],
    indirect=True,
)
def test_trackedit_widgets(
    viewer_and_trackedit, timelapse_mock_data, request
):  # noqa: F811
    """Test UI interactions with the viewer and widgets"""
    viewer, track_edit = viewer_and_trackedit

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

    toAnnotateBox = track_edit.AnnotationWidget.todo_box
    assert toAnnotateBox is not None, "ToAnnotateBox not found"

    TV = track_edit.tracksviewer

    check_selection(TV)
    check_time_box(time_box)
    check_editing(TV, editing_menu)
    check_red_flag_box(TV, red_flag_box)
    check_division_box(division_box)
    check_annotation(toAnnotateBox)

    check_export(navigation_widget)

    if request.config.getoption("--show-napari-viewer"):
        napari.run()


def check_selection(TV):
    """Check cell selection functionality"""

    # Test: Select single cell
    num_selected_before = len(TV.selected_nodes)
    TV.selected_nodes.add(3000009, append=False)
    num_selected_after = len(TV.selected_nodes)
    TV.selected_nodes.reset()
    assert num_selected_after == num_selected_before + 1, "Cell selection failed"

    # Test: Select multiple cells
    num_selected_before = len(TV.selected_nodes)
    TV.selected_nodes.add(3000006, append=True)
    TV.selected_nodes.add(3000009, append=True)
    num_selected_after = len(TV.selected_nodes)
    TV.selected_nodes.reset()
    assert (
        num_selected_after == num_selected_before + 2
    ), "Multiple cell selection failed"


def check_time_box(time_box):
    """Check time box functionality"""

    time_box.time_input.setText(str(19))
    time_box.on_time_input_entered()

    time_box.time_input.setText(str(2))
    time_box.on_time_input_entered()

    time_box.press_next()
    time_box.press_prev()


def check_editing(TV, editing_menu):
    """Check track editing functionality"""
    # Test: delete cell and undo
    TV.selected_nodes.add(3000009, append=False)
    TV.delete_node()
    TV.undo()

    # # Test: redo and undo deletion
    TV.redo()
    TV.undo()

    # Function to handle dialog
    def handle_dialog():
        print("handling dialog")
        for widget in QApplication.topLevelWidgets():
            if widget.windowTitle() == "Delete existing edge?":
                for button in widget.findChildren(QPushButton):
                    if button.text() == "&OK":
                        button.click()
                        break

    # # Test: break/add edge
    TV.selected_nodes.add(2000009, append=False)
    TV.selected_nodes.add(3000010, append=True)
    QTimer.singleShot(100, handle_dialog)  # handle popup window
    TV.create_edge()
    print("dialog correctly handled")

    # # Test: add node
    editing_menu.click_on_hierarchy_cell(3000012)
    editing_menu.add_cell_from_button()
    TV.undo()

    # # Test: duplicate node
    editing_menu.click_on_hierarchy_cell(3000012)
    editing_menu.duplicate_cell_id_input.setText(str(3000012))
    editing_menu.duplicate_time_input.setText(str(1))
    editing_menu.duplicate_cell_from_button()
    TV.undo()


def check_red_flag_box(TV, red_flag_box):
    """Check red flag box functionality"""

    # remove an extra node > create two extra red flags
    TV.selected_nodes.add(3000011, append=False)
    TV.delete_node()

    assert len(red_flag_box.databasehandler.red_flags) == 3

    TV.undo()

    assert len(red_flag_box.databasehandler.red_flags) == 1

    TV.redo()

    assert len(red_flag_box.databasehandler.red_flags) == 3

    red_flag_box.goto_red_flag()
    red_flag_box.go_to_next_red_flag()
    red_flag_box.go_to_prev_red_flag()

    while len(red_flag_box.databasehandler.red_flags) > 0:
        red_flag_box.ignore_red_flag()

    assert len(red_flag_box.databasehandler.red_flags) == 0

    red_flag_box.goto_red_flag()
    red_flag_box.go_to_next_red_flag()
    red_flag_box.go_to_prev_red_flag()


def check_division_box(division_box):
    """Check division box functionality"""
    division_box.goto_division()
    division_box.go_to_next_division()
    division_box.go_to_prev_division()


def check_annotation(toAnnotateBox):
    """Check annotation functionality"""

    toAnnotateBox.goto_annotation()
    toAnnotateBox.on_hair_clicked()
    toAnnotateBox.on_support_clicked()
    toAnnotateBox.on_mantle_clicked()

    toAnnotateBox.go_to_next_annotation()
    toAnnotateBox.go_to_prev_annotation()


def check_export(navigation_widget):
    """Check export functionality"""
    navigation_widget.export_btn.click()
