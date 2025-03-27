from pathlib import Path
from typing import Callable
from unittest.mock import patch

import napari
import pytest
from PyQt5.QtWidgets import QMessageBox
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
        tmax=5,
        scale=(1, 1, 1),
        allow_overwrite=True,
        time_chunk_length=3,
        time_chunk_overlap=1,
        imaging_zarr_file="",
        imaging_channel="",
        viewer=viewer,
    )

    return viewer, track_edit


@pytest.mark.parametrize(
    "timelapse_mock_data",  # noqa: F811
    [
        {"length": 4, "size": 64, "n_dim": 3},
    ],
    indirect=True,
)
def test_trackedit_widgets(
    viewer_and_trackedit, timelapse_mock_data, request  # noqa: F811
):
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

    toAnnotateBox = track_edit.AnnotationWidget.toannotate_box
    assert toAnnotateBox is not None, "ToAnnotateBox not found"

    TV = track_edit.tracksviewer

    check_selection(TV)
    check_time_box(time_box)
    check_editing(TV, editing_menu)
    check_red_flag_box(TV, red_flag_box, time_box)
    check_division_box(division_box)
    check_annotation(toAnnotateBox)
    check_export(navigation_widget)

    if request.config.getoption("--show-napari-viewer"):
        napari.run()


def check_selection(TV):
    """Check cell selection functionality"""

    # Test: Select single cell
    num_selected_before = len(TV.selected_nodes)
    TV.selected_nodes.add(2000009, append=False)
    num_selected_after = len(TV.selected_nodes)
    TV.selected_nodes.reset()
    assert num_selected_after == num_selected_before + 1, "Cell selection failed"

    # Test: Select multiple cells
    num_selected_before = len(TV.selected_nodes)
    TV.selected_nodes.add(2000006, append=True)
    TV.selected_nodes.add(2000009, append=True)
    num_selected_after = len(TV.selected_nodes)
    TV.selected_nodes.reset()
    assert (
        num_selected_after == num_selected_before + 2
    ), "Multiple cell selection failed"


def check_time_box(time_box):
    """Check time box functionality"""

    time_box.time_input.setText(str(4))
    time_box.on_time_input_entered()

    time_box.time_input.setText(str(0))
    time_box.on_time_input_entered()

    time_box.press_next()
    time_box.press_prev()


def check_editing(TV, editing_menu):
    """Check track editing functionality"""
    # # Test: delete cell and undo
    # Note: this is a node in the last frame of this timewindow, so the edge in the
    # next window will be removed, which is not recovered by undo. So we have 2 red
    # flags, one from the broken edge, one for the not-recovered on the next window
    TV.selected_nodes.add(2000009, append=False)
    TV.delete_node()
    TV.undo()

    # # Test: redo and undo deletion
    TV.redo()
    TV.undo()

    # Test: break/add edge with mocked dialog
    with patch.object(QMessageBox, "exec_", return_value=QMessageBox.Ok):
        TV.selected_nodes.add(1000009, append=False)
        TV.selected_nodes.add(2000010, append=True)
        TV.create_edge()

    # # Test: add node
    editing_menu.click_on_hierarchy_cell(2000012)
    editing_menu.add_cell_from_button()
    TV.undo()

    # # Test: duplicate node
    editing_menu.click_on_hierarchy_cell(2000012)
    editing_menu.duplicate_cell_id_input.setText(str(2000012))
    editing_menu.duplicate_time_input.setText(str(2))  # original time was 1
    editing_menu.duplicate_cell_from_button()
    TV.undo()

    # # Test: duplicate node that already exists
    editing_menu.click_on_hierarchy_cell(2000012)
    editing_menu.duplicate_cell_id_input.setText(str(2000012))
    editing_menu.duplicate_time_input.setText(str(1))  # original time was 1
    editing_menu.duplicate_cell_from_button()


def check_red_flag_box(TV, red_flag_box, time_box):
    """Check red flag box functionality"""

    # move to time 1
    time_box.time_input.setText(str(1))
    time_box.on_time_input_entered()

    # remove an extra node > create two extra red flags
    TV.selected_nodes.add(2000011, append=False)
    TV.delete_node()

    assert len(red_flag_box.databasehandler.red_flags) == 1

    TV.undo()

    assert len(red_flag_box.databasehandler.red_flags) == 0

    TV.redo()

    assert len(red_flag_box.databasehandler.red_flags) == 1

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

    toAnnotateBox.goto_annotation()
    toAnnotateBox.go_to_next_annotation()
    toAnnotateBox.go_to_prev_annotation()


def check_export(navigation_widget):
    """Check export functionality"""
    navigation_widget.export_btn.click()
