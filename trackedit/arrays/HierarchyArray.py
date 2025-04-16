from typing import Tuple, Union

import numpy as np
import sqlalchemy as sqla
from sqlalchemy import func
from sqlalchemy.orm import Session
from ultrack.config import MainConfig
from ultrack.core.database import NodeDB
from trackedit.utils.utils import apply_filters

class HierarchyArray:
    def __init__(
        self,
        config: MainConfig,
        dtype: np.dtype = np.int32,
        extra_filters: list[sqla.Column] = [],
    ):
        """Create an array that directly visualizes the segments in the ultrack database.

        Parameters
        ----------
        config : MainConfig
            Configuration file of Ultrack.
        dtype : np.dtype
            Data type of the array.
        extra_filters : list[sqla.Column]
            Additional filters to apply to the query, e.g. [NodeDB.x < 300]
        """

        self.config = config
        self.shape = tuple(config.data_config.metadata["shape"])  # (t,(z),y,x)
        self.dtype = dtype
        self.t_max = self.shape[0]
        self.ndim = len(self.shape)
        self.array = np.zeros(self.shape[1:], dtype=self.dtype)
        self.time_window = [0, self.shape[0]]
        self.extra_filters = extra_filters

        self.database_path = config.data_config.database_path
        self.minmax = self.find_min_max_volume_entire_dataset()
        self.volume = self.minmax.mean().astype(int)

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
        # print('indexing in getitem:',indexing)

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

        chunk_time = time + self.time_window[0]
        self.fill_array(
            time=chunk_time,
        )

        return self.array[volume_slicing]

    def fill_array(
        self,
        time: int,
    ) -> None:
        """Paint all segments of specific time point which volume is bigger than self.volume
        Parameters
        ----------
        time : int
            time point to paint the segments
        """

        volume = float(self.volume) if hasattr(self.volume, 'item') else self.volume
        filters = [NodeDB.t == time, NodeDB.area < volume] + self.extra_filters

        engine = sqla.create_engine(self.database_path)
        self.array.fill(0)

        with Session(engine) as session:
            query = session.query(NodeDB.pickle, NodeDB.id, NodeDB.hier_parent_id, NodeDB.area)
            query = apply_filters(query, filters)            
            query = list(query)

            if len(query) == 0:
                return

            nodes, node_ids, parent_ids, areas = zip(*query)

            node_ids_set = set(node_ids)  # faster lookup

            count = 0
            for i in range(len(nodes)):
                # only paint top-most level of hierarchy
                if parent_ids[i] not in node_ids_set:
                    nodes[i].paint_buffer(
                        self.array, value=node_ids[i], include_time=False
                    )
                    count += 1


    def get_tp_num_pixels(
        self,
        timeStart: int,
        timeStop: int,
    ) -> list:
        """Gets a list of number of pixels of all segments range of time points (timeStart to timeStop)
        Parameters
        ----------
        timeStart : int
        timeStop : int
        Returns
        -------
        num_pix_list : list
            list with all num_pixels for timeStart to timeStop
        """
        engine = sqla.create_engine(self.database_path)
        num_pix_list = []
        with Session(engine) as session:
            query = list(
                session.query(NodeDB.area).where(NodeDB.t.between(timeStart, timeStop))
            )
            for num_pix in query:
                num_pix_list.append(int(np.array(num_pix)))
        return num_pix_list

    def find_min_max_volume_entire_dataset(self):
        """Find minimum and maximum segment volume for ALL time point

        Returns
        -------
        np.array : np.array
            array with two elements: [min_volume, max_volume]
        """
        engine = sqla.create_engine(self.database_path)
        with Session(engine) as session:
            max_vol = (
                session.query(func.max(NodeDB.area))
                .where(NodeDB.t.between(0, self.t_max))
                .scalar()
            )
            min_vol = (
                session.query(func.min(NodeDB.area))
                .where(NodeDB.t.between(0, self.t_max))
                .scalar()
            )

        return np.array([min_vol, max_vol], dtype=int)

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
        self.shape = (
            self.time_window[1] - self.time_window[0],
            *self.shape[1:],
        )  # shape=tuple in UA, but list in DBA
        self.t_max = self.shape[0]
        self.ndim = len(self.shape)
        self.array = np.zeros(self.shape[1:], dtype=self.dtype)
