from typing import Tuple

import dask.array as da


class SimpleImageArray:
    """A simple wrapper for zarr/dask image data that can be used with napari.ImageLayer"""

    def __init__(
        self,
        imaging_zarr_file: str,
        channel: str = "0/4/0/0",
        time_window: Tuple[int, int] = (0, 104),
    ):
        """
        Initialize the image array from a zarr file.

        Args:
            imaging_zarr_file: Path to the zarr file
            channel: Channel path within the zarr file (default: '0/4/0/0')
        """
        # Load the full stack using dask
        self._full_stack = da.from_zarr(imaging_zarr_file, component=channel)
        # Initialize time window to full range
        self._time_window = time_window

        # Set initial stack based on full time window
        self._update_stack()

    def _update_stack(self):
        """Update the stack based on current time window"""
        self._stack = self._full_stack[self._time_window[0] : self._time_window[1]]

    @property
    def nuclear(self) -> da.Array:
        """Get the nuclear channel data"""
        return self._stack[:, 0]

    @property
    def membrane(self) -> da.Array:
        """Get the membrane channel data"""
        return self._stack[:, 1]

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of a single channel (identical for nuclear and membrane)"""
        return self._stack[:, 0].shape

    @property
    def time_window(self) -> Tuple[int, int]:
        """Get current time window"""
        return tuple(self._time_window)

    def set_time_window(self, window: Tuple[int, int]) -> None:
        """
        Set time window and update stack accordingly

        Args:
            window: Tuple of (start_time, end_time)
        """
        if not isinstance(window, tuple):
            raise ValueError("Time window must be a tuple of (start_time, end_time)")
        if len(window) != 2:
            raise ValueError("Time window must contain exactly two values")

        start, end = window
        if start < 0 or end > self._full_stack.shape[0] or start >= end:
            raise ValueError(
                f"Invalid time window. Must be between 0 and {self._full_stack.shape[0]}"
            )

        self._time_window = list(window)
        self._update_stack()
