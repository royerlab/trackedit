from typing import List, Optional, Tuple

import dask.array as da


class SimpleImageArray:
    """A simple wrapper for zarr/dask image data that can be used with napari.ImageLayer"""

    def __init__(
        self,
        imaging_zarr_file: str,
        channel: str = "0/4/0/0",
        time_window: Tuple[int, int] = (0, 104),
        image_z_slice: int = None,
        imaging_layer_names: Optional[List[str]] = None,
    ):
        """
        Initialize the image array from a zarr file.

        Args:
            imaging_zarr_file: Path to the zarr file
            channel: Channel path within the zarr file (default: '0/4/0/0')
            time_window: Time window for the image stack
            image_z_slice: Optional z-slice to extract
            imaging_layer_names: Names for imaging channels. If None, defaults to ['nuclear', 'membrane'] for 2 channels
        """
        # Load the full stack using dask
        self._full_stack = da.from_zarr(imaging_zarr_file, component=channel)
        if image_z_slice is not None:
            old_shape = self._full_stack.shape
            self._full_stack = self._full_stack[:, :, image_z_slice, :, :]
            print("image stack resliced from", old_shape, "to", self._full_stack.shape)
        # Initialize time window to full range
        self._time_window = time_window

        # Set initial stack based on full time window
        self._update_stack()

        # Detect number of channels and set layer names
        self.n_channels = self._stack.shape[1] if len(self._stack.shape) > 1 else 1

        # Set layer names with backward compatibility
        if imaging_layer_names is None:
            if self.n_channels == 2:
                self.layer_names = ["nuclear", "membrane"]
            else:
                self.layer_names = [f"channel_{i}" for i in range(self.n_channels)]
        else:
            if len(imaging_layer_names) != self.n_channels:
                raise ValueError(
                    f"Number of layer names ({len(imaging_layer_names)}) "
                    f"doesn't match number of channels ({self.n_channels})"
                )
            self.layer_names = imaging_layer_names.copy()

    def _update_stack(self):
        """Update the stack based on current time window"""
        self._stack = self._full_stack[self._time_window[0] : self._time_window[1]]

    def get_channel_data(self, channel_idx: int) -> da.Array:
        """Get data for a specific channel by index"""
        if channel_idx >= self.n_channels:
            raise ValueError(
                f"Channel index {channel_idx} >= number of channels ({self.n_channels})"
            )
        return self._stack[:, channel_idx]

    def get_channel_by_name(self, name: str) -> da.Array:
        """Get channel data by name"""
        if name not in self.layer_names:
            raise ValueError(f"Channel name '{name}' not found in {self.layer_names}")
        idx = self.layer_names.index(name)
        return self.get_channel_data(idx)

    @property
    def nuclear(self) -> da.Array:
        """Get the nuclear channel data (backward compatibility)"""
        return self.get_channel_by_name("nuclear")

    @property
    def membrane(self) -> da.Array:
        """Get the membrane channel data (backward compatibility)"""
        return self.get_channel_by_name("membrane")

    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of a single channel"""
        return self.get_channel_data(0).shape

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
