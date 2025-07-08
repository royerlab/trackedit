from pathlib import Path
from typing import Callable

import napari
import numpy as np
import pytest
import zarr
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


def create_mock_imaging_zarr(
    working_directory: Path, tmax: int = 4, size: int = 64, n_dim: int = 3
) -> str:
    """
    Create a mock zarr file with the correct dimensions for testing imaging data.

    Args:
        working_directory: Directory to create the zarr file in
        tmax: Number of time points
        size: Size of each dimension (assumes cubic data)
        n_dim: Number of spatial dimensions (2 or 3)

    Returns:
        Path to the created zarr file
    """
    # Create zarr file path
    zarr_path = working_directory / "mock_imaging.zarr"

    # Create the zarr group
    root = zarr.open_group(str(zarr_path), mode="w")

    # Create the nested structure: 0/4/0/0
    # This matches the default channel path used in the code
    group_0 = root.create_group("0")
    group_4 = group_0.create_group("4")
    group_0_inner = group_4.create_group("0")

    # Determine shape based on dimensions
    if n_dim == 3:
        # 3D data: (T, Z, Y, X, C) where C=2 for nuclear and membrane channels
        shape = (tmax, size, size, size, 2)
    else:
        # 2D data: (T, Y, X, C) where C=2 for nuclear and membrane channels
        shape = (tmax, size, size, 2)

    # Create empty array with zeros
    data = np.zeros(shape, dtype=np.uint16)

    # Store in the zarr array
    group_0_inner.create_dataset(
        "0", data=data, chunks=(1, size // 4, size // 4, size // 4, 2)
    )

    return str(zarr_path)


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

    # Create mock imaging zarr file
    imaging_zarr_file = create_mock_imaging_zarr(
        working_directory=working_directory, tmax=4, size=64, n_dim=3
    )
    imaging_channel = "0/4/0/0"

    track_edit = run_trackedit(
        working_directory=working_directory,
        db_filename="data.db",
        tmax=tmax,
        scale=scale,
        allow_overwrite=True,
        time_chunk_length=3,
        time_chunk_overlap=1,
        viewer=viewer,
        imaging_zarr_file=imaging_zarr_file,
        imaging_channel=imaging_channel,
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

    # Get the navigation widget
    navigation_widget = track_edit.NavigationWidget
    assert navigation_widget is not None, "NavigationWidget not found"

    # Navigate to next chunk
    navigation_widget.change_chunk.emit("next")

    # The data should be different after navigation (different time window)
    # Note: Since we're using zeros, the data will be identical, but the time window should change
    assert track_edit.databasehandler.imagingArray.time_window != (
        0,
        3,
    ), "Time window should change after navigation"

    # Navigate back to previous chunk
    navigation_widget.change_chunk.emit("prev")

    # Check that we're back to the original time window
    assert track_edit.databasehandler.imagingArray.time_window == (
        0,
        3,
    ), "Should return to original time window"

    # Test navigation to a specific frame
    navigation_widget.goto_frame.emit(2)  # Go to frame 2

    if request.config.getoption("--show-napari-viewer"):
        napari.run()
