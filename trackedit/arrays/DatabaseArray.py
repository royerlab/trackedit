from pathlib import Path
from typing import Tuple, Union

import numpy as np
import sqlalchemy as sqla
from sqlalchemy import Column
from sqlalchemy.orm import Session
from ultrack.core.database import NodeDB

from trackedit.utils.utils import apply_filters


class DatabaseArray:
    def __init__(
        self,
        database_path: Path,
        shape: Tuple[int, ...],  # (t,(z),y,x)
        time_window: tuple,
        color_by_field: Column = NodeDB.id,
        dtype: np.dtype = np.int32,
        current_time: int = np.nan,
        extra_filters: list[sqla.Column] = [],
    ):
        """Create an array that directly visualizes the segments in the ultrack database.

        Parameters
        ----------
        database_path : Path
            Path to the ultrack database.
        shape : Tuple[int, ...]
            Shape of the array, e.g. (t, z, y, x)
        time_window : Tuple[int, int]
            Time window of the array, e.g. (0, 100)
        color_by_field : Column
            Column to color the array by, e.g. NodeDB.id
        dtype : np.dtype
            Data type of the array, e.g. np.int32
        current_time : int
            Current time point of the array, e.g. 0
        extra_filters : list[sqla.Column]
            Additional filters to apply to the query, e.g. [NodeDB.x < 300]
        """
        self.database_path = database_path
        self.shape = shape
        self.dtype = dtype
        self.current_time = current_time
        self.time_window = time_window
        self.color_by_field = color_by_field
        self.extra_filters = extra_filters

        self.ndim = len(self.shape)
        self.array = np.zeros(self.shape[1:], dtype=self.dtype)

    def __getitem__(
        self,
        indexing: Union[Tuple[Union[int, slice]], int, slice],
    ) -> np.ndarray:
        """Indexing the ultrack-array

        Parameters
        ----------
        indexing : Tuple or Array

        Returns
        -------
        array : numpy array
            array with painted segments
        """

        if isinstance(indexing, tuple):
            time, volume_slicing = indexing[0], indexing[1:]
        else:  # if only 1 (time) is provided
            time = indexing
            volume_slicing = tuple()

        if isinstance(time, slice):  # if all time points are requested
            return np.stack(
                [
                    self.__getitem__((t,) + volume_slicing)
                    for t in range(*time.indices(self.shape[0]))
                ]
            )
        else:
            try:
                time = time.item()  # convert from numpy.int to int
            except AttributeError:
                time = time

        time = time + self.time_window[0]

        if (time != self.current_time) or (
            time == 0
        ):  # always refill if time is 0, because napari regularly fetches time 0,
            # so sometimes the current time is not 0, but we still need to refill after an update
            self.current_time = time
            self.fill_array(
                time=time,
            )

        return self.array[volume_slicing]

    def __setitem__(
        self,
        indexing: Union[Tuple[Union[int, slice]], int, slice],
        value: Union[np.ndarray, int, float],
    ) -> None:
        print(
            "setting a value in DatabaseArray not allowed, all interaction goes via db"
        )

    def __array__(self, dtype=None, copy=True):
        if dtype is None:
            return np.asarray(self.array)
        return np.asarray(self.array, dtype=dtype)

    def set_time_window(self, time_window: Tuple[int, int]) -> None:
        """Set the time window of the array.

        Parameters
        ----------
        time_window : Tuple[int, int]
            Time window of the array.

        Returns
        -------
        None
        """
        self.time_window = time_window
        self.shape[0] = self.time_window[1] - self.time_window[0]

    def force_refill(self):
        """Force the array to be filled with the current time point."""
        self.fill_array(self.current_time)

    def fill_array(
        self,
        time: int,
    ) -> None:
        """
        Fill the array with the selected segments from the database, for one time point.

        Parameters
        ----------
        time : int
            Time point to fill the array
        extra_filters : list[sqla.Column]
            Additional filters to apply to the query, e.g. [NodeDB.x < 300]

        Returns
        -------
        None
        """

        filters = [NodeDB.t == time, NodeDB.selected] + self.extra_filters

        engine = sqla.create_engine(self.database_path)
        self.array.fill(0)

        with Session(engine) as session:
            query = session.query(self.color_by_field, NodeDB.pickle)
            query = apply_filters(query, filters)
            query = list(query)

            if len(query) == 0:
                return

            for idx, q in enumerate(query):
                q[1].paint_buffer(self.array, value=q[0], include_time=False)
