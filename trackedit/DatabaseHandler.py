import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import toml
from motile_toolbox.candidate_graph import NodeAttr
from sqlalchemy import create_engine, inspect, text
from ultrack.config import MainConfig
from ultrack.core.database import (
    Column,
    Integer,
    NodeDB,
    get_node_values,
    set_node_values,
)
from ultrack.core.export import tracks_layer_to_networkx, tracks_to_zarr
from ultrack.tracks.graph import add_track_ids_to_tracks_df

from trackedit.__about__ import __version__
from trackedit.arrays.DatabaseArray import DatabaseArray
from trackedit.arrays.ImagingArray import SimpleImageArray
from trackedit.utils.red_flag_funcs import (
    combine_red_flags,
    filter_red_flags_at_edge,
    find_all_starts_and_ends,
    find_overlapping_cells,
)
from trackedit.utils.utils import (
    annotations_to_zarr,
    remove_nonexisting_parents,
    solution_dataframe_from_sql_with_tmax,
)

NodeDB.generic = Column(Integer, default=-1)


class DatabaseHandler:
    def __init__(
        self,
        db_filename_old: str,
        working_directory: Path,
        Tmax: int,
        scale: tuple,
        name: str,
        annotation_mapping: dict = None,
        time_chunk: int = 0,
        time_chunk_length: int = 105,
        time_chunk_overlap: int = 5,
        allow_overwrite: bool = False,
        work_in_existing_db: bool = False,
        imaging_zarr_file: str = None,
        imaging_channel: str = None,
        image_z_slice: int = None,
        image_translate: tuple = None,
        coordinate_filters: list = None,
        default_start_annotation: int = None,  # Make it optional
        imaging_layer_names: list = None,
        flag_remove_red_flags_at_edge: bool = False,
        remove_red_flags_at_edge_threshold: int = 10,
    ):

        # inputs
        self.db_filename_old = db_filename_old
        self.working_directory = working_directory
        self.Tmax = Tmax
        self.scale = scale
        self.name = name
        self.allow_overwrite = allow_overwrite
        self.work_in_existing_db = work_in_existing_db
        self.time_chunk = time_chunk
        self.time_chunk_length = time_chunk_length
        self.time_chunk_overlap = time_chunk_overlap
        self.imaging_zarr_file = imaging_zarr_file
        self.imaging_channel = imaging_channel
        self.image_z_slice = image_z_slice
        self.image_translate = image_translate
        self.imaging_flag = (
            True if self.imaging_zarr_file and self.imaging_zarr_file != "" else False
        )
        self.coordinate_filters = coordinate_filters
        self.imaging_layer_names = imaging_layer_names
        self.flag_remove_red_flags_at_edge = flag_remove_red_flags_at_edge
        self.remove_red_flags_at_edge_threshold = remove_red_flags_at_edge_threshold

        # Filenames / directories
        self.extension_string = ""
        (
            self.db_filename_old,
            self.db_filename_new,
            self.log_filename_new,
        ) = self.copy_database(
            self.working_directory,
            self.db_filename_old,
            allow_overwrite=self.allow_overwrite,
            work_in_existing_db=self.work_in_existing_db,
        )
        self.db_path_new = f"sqlite:///{self.working_directory/self.db_filename_new}"
        self.log_file = self.initialize_logfile(
            self.log_filename_new, self.work_in_existing_db
        )

        # get data shape from metadata.toml
        self.config_adjusted = self.initialize_config()
        metadata_path = self.working_directory / "metadata.toml"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_loaded = toml.load(f)
                print("metadata_loaded:", metadata_loaded)
            self.config_adjusted.data_config.metadata.update(metadata_loaded)
        else:
            print("No metadata.toml file found")
        self.data_shape_full = self.config_adjusted.data_config.metadata["shape"]
        if self.Tmax > self.data_shape_full[0]:
            self.Tmax = self.data_shape_full[0]
        else:
            self.data_shape_full[0] = self.Tmax

        # check ndim and scale
        self.ndim = len(
            self.data_shape_full
        )  # number of dimensions of the data, 4 for 3D+time, 3 for 2D+time
        if self.ndim != len(self.scale) + 1:
            raise ValueError(
                f"Expected scale with {self.ndim-1} values, (Z)YX, but scale has {len(self.scale)} values."
            )
        if self.ndim == 4:
            self.z_scale = self.scale[0]
            self.y_scale = self.scale[1]
            self.x_scale = self.scale[2]
        elif self.ndim == 3:
            self.z_scale = None
            self.y_scale = self.scale[0]
            self.x_scale = self.scale[1]
        else:
            raise ValueError(f"Database should be 2D or 3D, not {self.ndim-1}D")

        # change initial chunk depending on data shape
        if self.data_shape_full[0] < self.time_chunk_length:
            self.time_chunk_length = self.data_shape_full[0]
            self.time_chunk_overlap = 0

        # calculate time chunk
        (
            self.time_window,
            self.time_chunk_starts,
            self.num_time_chunks,
        ) = self.calc_time_window()
        self.data_shape_chunk = self.data_shape_full.copy()
        self.data_shape_chunk[0] = self.time_chunk_length

        # Store the default annotation value, fall back to NodeDB.generic default if not provided
        self.default_start_annotation = (
            default_start_annotation
            if default_start_annotation is not None
            else NodeDB.generic.default.arg
        )

        self.add_missing_columns_to_db()

        # DatabaseArray()
        self.segments = DatabaseArray(
            database_path=self.db_path_new,
            shape=self.data_shape_chunk,
            time_window=self.time_window,
            color_by_field=NodeDB.id,
            coordinate_filters=self.coordinate_filters,
        )
        self.annotArray = DatabaseArray(
            database_path=self.db_path_new,
            shape=self.data_shape_chunk,
            time_window=self.time_window,
            color_by_field=NodeDB.generic,
            coordinate_filters=self.coordinate_filters,
        )
        self.check_zarr_existance()
        if self.imaging_flag:
            self.imagingArray = SimpleImageArray(
                imaging_zarr_file=self.imaging_zarr_file,
                channel=self.imaging_channel,
                time_window=self.time_window,
                image_z_slice=self.image_z_slice,
                imaging_layer_names=self.imaging_layer_names,
            )
        self.df_full = self.db_to_df(entire_database=True)

        # ToDo: df_full might be very large for large datasets, but annotation/redflags/division need it
        self.nxgraph = self.df_to_nxgraph()
        self.red_flags_ignore_list = self._load_red_flags_ignore_list()
        self.recompute_red_flags()
        self.toannotate = self.find_all_toannotate()
        self.divisions = self.find_all_divisions()
        self.log(
            f"Start annotation session - TrackEdit v{__version__} ({datetime.now().replace(microsecond=0)})"
        )
        self.log(
            f"Parameters: Tmax: {self.Tmax}, working_directory: {self.working_directory}, "
            f"db_filename: {self.db_filename_new}"
        )

        # print red flag summary
        print("\nRed flag summary:")
        print(f"  Total red flags: {len(self.red_flags)}")
        for event, count in self.red_flags.event.value_counts().items():
            print(f"    {event}: {count}")

        # Default label for unlabeled cells
        default_annotation = {
            self.default_start_annotation: {  # Use the instance variable
                "name": "none",
                "color": [0.5, 0.5, 0.5, 1.0],  # gray
            }
        }

        # If no custom mapping provided, use the original default mapping
        if annotation_mapping is None:
            annotation_mapping = {
                1: {"name": "hair", "color": [0.0, 1.0, 0.0, 1.0]},  # green
                2: {"name": "support", "color": [1.0, 0.1, 0.6, 1.0]},  # pink
                3: {"name": "mantle", "color": [0.0, 0.0, 0.9, 1.0]},  # blue
            }

        # Combine default label with custom mapping
        self.annotation_mapping_dict = default_annotation | annotation_mapping
        self.check_annotation_mapping()

    def initialize_logfile(self, log_filename_new, work_in_existing_db):
        """Initialize the logger with a file path. Raises an error if the file already exists."""

        log_file_path = self.working_directory / log_filename_new

        if (os.path.exists(log_file_path)) & (not work_in_existing_db):
            if not self.allow_overwrite:
                raise FileExistsError(
                    f"Log file '{log_file_path}' already exists. Choose a different file or delete the existing one."
                )
            else:
                open(log_file_path, "w").close()  # Clear the file if it exists

        print("old database:", self.db_filename_old)
        print("new database:", self.db_filename_new)
        print("new logfile:", log_filename_new)

        return log_file_path

    def log(self, message, is_header=True):
        """Append a message to the log file."""
        with open(self.log_file, "a") as log:
            time_stamp = f"[{datetime.now().replace(microsecond=0)}]"
            if is_header:
                log.write(message + "\n")
            else:
                log.write("\t\t" + time_stamp + " " + message + "\n")

    def initialize_config(self):
        # import db filename properly into an Ultrack config, neccesary for chaning values in database
        config_adjusted = MainConfig()
        config_adjusted.data_config.working_dir = self.working_directory
        config_adjusted.data_config.database_file_name = self.db_filename_new
        return config_adjusted

    def copy_database(
        self, working_directory, db_filename_old, allow_overwrite, work_in_existing_db
    ):
        """
        Copy the database to a new versioned filename.
        Ensures the new filename does not already exist before copying.
        """
        if work_in_existing_db:
            (
                db_filename_old,
                db_filename_new,
                log_filename_new,
            ) = self.get_same_db_filename(db_filename_old)
        else:
            (
                db_filename_old,
                db_filename_new,
                log_filename_new,
            ) = self.get_next_db_filename(db_filename_old)

        # Create full paths
        old_db_path = Path(working_directory) / db_filename_old
        new_db_path = Path(working_directory) / db_filename_new

        # Check if the old database file exists
        if not old_db_path.exists():
            raise FileNotFoundError(
                f"Error: {db_filename_old} not found in the working directory."
            )

        # Check if the new database file already exists
        if not work_in_existing_db:
            if new_db_path.exists() & (not allow_overwrite):
                raise FileExistsError(
                    f"Error: {db_filename_new} already exists. "
                    "Set 'allow_overwrite' to 'True', or "
                    "choose a different database filename."
                    "('latest' assures that the latest database is used)"
                )
            else:
                shutil.copy(old_db_path, new_db_path)
                print(f"Database copied to: {new_db_path}")
        else:  # work in existing db
            if not new_db_path.exists():
                raise FileExistsError(
                    f"Error: {db_filename_new} does not exist. "
                    "Set filename to a database that exists, "
                    "or set work_in_existing_db to False."
                )

        return db_filename_old, db_filename_new, log_filename_new

    def add_missing_columns_to_db(self):
        engine = create_engine(self.config_adjusted.data_config.database_path)
        inspector = inspect(engine)

        expected_columns = self.get_expected_columns(engine)
        existing_columns = [col["name"] for col in inspector.get_columns("nodes")]

        for col_name, col_definition in expected_columns.items():
            if col_name not in existing_columns:
                alter_stmt = (
                    f"ALTER TABLE nodes ADD COLUMN {col_name} {col_definition};"
                )
                with engine.begin() as conn:  # This starts a transaction.
                    conn.execute(text(alter_stmt))
                print(f"Added missing column to db: {col_name}")

    def get_expected_columns(self, engine):
        expected_columns = {}
        for column in NodeDB.__table__.columns:
            # Compile the column type to its SQL representation.
            col_type_str = column.type.compile(dialect=engine.dialect)

            # Try to extract a default value if one is specified.
            default_value = None
            if column.default is not None and hasattr(column.default, "arg"):
                default_value = column.default.arg

            # Build the definition string.
            col_definition = col_type_str
            if default_value is not None:
                col_definition += f" DEFAULT {default_value}"

            expected_columns[column.name] = col_definition

        # Add the new generic column using the custom default annotation value
        expected_columns["generic"] = f"INTEGER DEFAULT {self.default_start_annotation}"

        return expected_columns

    def get_next_db_filename(self, old_filename):
        """
        Generate the next version of a database filename.

        - If the filename is 'data.db', it returns 'data_v1.db'.
        - If the filename is 'data_v3.db', it returns 'data_v4.db'.
        - If the filename is 'latest', it returns 'data_vN.db', where N is the highest version number.

        Returns
        -------
        tuple
            (old_filename, new_db_filename, new_log_filename)
        """
        if old_filename == "latest":
            # Find all database files in the working directory that match the pattern data_v*.db
            pattern = re.compile(r"data_v(\d+)\.db$")
            version_files = []

            for file in self.working_directory.glob("data_v*.db"):
                match = pattern.match(file.name)
                if match:
                    version_files.append((int(match.group(1)), file.name))

            # If no versioned files found, start with v1 and use data.db as old file
            if not version_files:
                version_number = 1
                old_filename = "data.db"
            else:
                # Sort by version number and get the highest one
                version_files.sort(key=lambda x: x[0])
                version_number = version_files[-1][0] + 1
                old_filename = version_files[-1][1]

            base_name = "data"
            ext = ".db"
        else:
            # Existing logic for other cases
            name, ext = os.path.splitext(old_filename)
            match = re.match(r"^(.*)_v(\d+)$", name)

            if match:
                base_name = match.group(1)  # Extract name before "_v"
                version_number = int(match.group(2)) + 1  # Increment version
            else:
                base_name = name  # No versioning found, use the original name
                version_number = 1  # Start with version 1

        # Create the new filename
        self.extension_string = (
            f"v{version_number}"  # "_v3" (saved for export function)
        )
        db_filename_new = f"{base_name}_{self.extension_string}{ext}"
        log_filename_new = f"data_v{version_number}_changelog.txt"
        return old_filename, db_filename_new, log_filename_new

    def get_same_db_filename(self, old_filename):
        """
        Generate the same database filename, handling the 'latest' special case.
        """
        if old_filename == "latest":
            # Find all database files in the working directory that match the pattern data_v*.db
            pattern = re.compile(r"data_v(\d+)\.db$")
            version_files = []

            for file in self.working_directory.glob("data_v*.db"):
                match = pattern.match(file.name)
                if match:
                    version_files.append((int(match.group(1)), file.name))

            # If no versioned files found, use data.db as the file
            if not version_files:
                old_filename = "data.db"
            else:
                # Sort by version number and get the highest one
                version_files.sort(key=lambda x: x[0])
                old_filename = version_files[-1][1]

        name, ext = os.path.splitext(old_filename)
        old_filename = name + ext
        db_filename_new = old_filename
        log_filename_new = f"{name}_changelog.txt"
        return old_filename, db_filename_new, log_filename_new

    def change_values(self, indices, field, values, log_header=None):
        """Change values in the database for one or multiple indices.

        Parameters
        ----------
        indices : int or list
            Single index or list of indices to change
        field : Column
            Database field to change
        value : int or list
            Value(s) to set. If a single value is provided, it will be applied to all indices.
            If a list is provided, it must match the length of indices.
        log_header : str, optional
            Optional header to prepend to log messages.
        """
        # Convert single index to list
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        else:
            indices = [int(i) for i in indices]

        if log_header is not None:
            self.log(log_header)

        # Handle value input
        if isinstance(values, (list, np.ndarray)):
            if len(values) != len(indices):
                raise ValueError(
                    f"Length of values ({len(values)}) must match length of indices ({len(indices)})"
                )
            values = [int(v) for v in values]
        else:
            values = int(values)
            values = [values] * len(indices)

        # Get old values for logging
        old_vals = get_node_values(
            self.config_adjusted.data_config, indices=indices, values=field
        )

        # Set new values
        set_node_values(
            self.config_adjusted.data_config, indices=indices, **{field.name: values}
        )

        # Update the dataframe
        self.df_full = self.db_to_df(entire_database=True)

        # Log changes
        for i in range(len(indices)):
            # handle different types of old_vals (list, np.ndarray, pd.Series)
            old_val = (
                old_vals[i]
                if isinstance(old_vals, (list, np.ndarray))
                else old_vals.iloc[i]
                if hasattr(old_vals, "iloc")
                else old_vals
            )
            message = f"db: setting {field.name}[id={indices[i]}] = {values[i]} (was {old_val})"
            self.log(message, is_header=False)

    def calc_time_window(self):
        time_chunk_starts = np.arange(
            0, self.Tmax, self.time_chunk_length - self.time_chunk_overlap
        )
        time_chunk_stops = np.array(
            [
                (s + self.time_chunk_length)
                if (s + self.time_chunk_length < self.Tmax)
                else self.Tmax
                for s in time_chunk_starts
            ]
        ).astype(int)

        if self.time_chunk >= len(time_chunk_starts):
            self.time_chunk = len(time_chunk_starts) - 1

        time_window = (
            int(time_chunk_starts[self.time_chunk]),
            int(time_chunk_stops[self.time_chunk]),
        )

        num_time_chunks = len(
            np.arange(0, self.Tmax, self.time_chunk_length - self.time_chunk_overlap)
        )

        return time_window, time_chunk_starts, num_time_chunks

    def set_time_chunk(self, time_chunk):
        if time_chunk >= self.num_time_chunks:
            raise ValueError(
                f"Time chunk {time_chunk} out of range. Maximum time chunk is {self.num_time_chunks-1}"
            )
        else:
            self.time_chunk = time_chunk
            (
                self.time_window,
                self.time_chunk_starts,
                self.num_time_chunks,
            ) = self.calc_time_window()
            self.segments.set_time_window(self.time_window)
            self.annotArray.set_time_window(self.time_window)
            if self.imaging_flag:
                self.imagingArray.set_time_window(self.time_window)
            self.nxgraph = self.df_to_nxgraph()

    def check_if_tmax_changed(self):
        """
        Checks if the max time in the database has changed.
        Called from NavigationWidget.check_if_tmax_changed()
        """
        if len(self.df_full) == 0:  # If dataframe is empty
            return False, self.Tmax

        current_max_time = self.df_full["t"].max()

        # If max time is less than current Tmax, update it
        if (
            current_max_time < self.Tmax - 1
        ):  # Tmax of 600 means 599 is the last timepoint
            self.Tmax = current_max_time
            (
                self.time_window,
                self.time_chunk_starts,
                self.num_time_chunks,
            ) = self.calc_time_window()
            return True, self.Tmax
        else:
            return False, self.Tmax

    def find_chunk_from_frame(self, frame):
        """Find the chunk index for a given frame. Since the chunks have an overlap, this function is not trivial"""
        chunk = np.where(frame >= self.time_chunk_starts)[0][-1]
        return chunk

    def db_to_df(
        self,
        entire_database: bool = False,
        include_parents: bool = True,
        include_node_ids: bool = True,
    ) -> pd.DataFrame:
        """Exports solution from database to pandas dataframe.

        Parameters
        ----------
        entire_database : bool
            Flag to include all time points in the database. By default, only the time window is included.
        include_parents : bool
            Flag to include parents track id for each track id.
        include_ids : bool
            Flag to include node ids for each unit.

        Returns
        -------
        pd.DataFrame
            Dataframe with columns: track_id, t, z, y, x
        """

        df = solution_dataframe_from_sql_with_tmax(
            self.db_path_new,
            tmax=self.Tmax,
            columns=(
                NodeDB.id,
                NodeDB.parent_id,
                NodeDB.t,
                NodeDB.z,
                NodeDB.y,
                NodeDB.x,
                NodeDB.area,
                NodeDB.generic,
            ),
        )

        if df.index.values.min() <= 0:
            raise ValueError(
                "Database contains nodes with node_ids <= 0. Please ensure all node IDs are positive integers."
            )

        if self.coordinate_filters is not None:
            for field, op, value in self.coordinate_filters:
                df = df[op(df[field.name], value)]

        df = remove_nonexisting_parents(df)
        df = add_track_ids_to_tracks_df(df)
        df.sort_values(by=["track_id", "t"], inplace=True)

        if not entire_database:
            if self.time_window is not None:
                min_time = self.time_window[0]
                max_time = self.time_window[1]
                df = df[(df.t >= min_time) & (df.t < max_time)].copy()
        else:
            df = df[df.t < self.Tmax].copy()

        df = self.remove_past_parents_from_df(df)

        if self.ndim == 4:
            columns = ["track_id", "t", "z", "y", "x"]
        elif self.ndim == 3:
            columns = ["track_id", "t", "y", "x"]
        else:
            raise ValueError(
                f"Expected dataset with 3 or 4 dimensions, T(Z)YX. Found {self.ndim}."
            )

        if include_node_ids:
            df.loc[:, "id"] = df.index
            columns.append("id")

        if include_parents:
            columns.append("parent_track_id")

            if include_node_ids:
                columns.append("parent_id")

        columns.append("generic")
        df = df[columns]
        return df

    def df_to_nxgraph(self) -> nx.DiGraph:
        """Exports solution from tracks dataframe to networkx graph.

        Returns
        nx.DiGraph
            Networkx graph.

        """
        # apply scale, only do this here to avoid scaling the original dataframe
        df_scaled = self.db_to_df()
        if self.ndim == 4:  # 4 for 3D+time, 3 for 2D+time
            df_scaled.loc[:, "z"] = df_scaled.z * self.z_scale  # apply scale
        df_scaled.loc[:, "y"] = df_scaled.y * self.y_scale  # apply scale
        df_scaled.loc[:, "x"] = df_scaled.x * self.x_scale  # apply scale

        nxgraph = tracks_layer_to_networkx(df_scaled)

        # add/modify attributes to fit motile viewer assumptions
        for node in nxgraph.nodes:
            nxgraph.nodes[node][NodeAttr.TRACK_ID.value] = int(
                nxgraph.nodes[node]["track_id"]
            )

        return nxgraph

    def remove_past_parents_from_df(self, df2):

        df2.loc[:, "t"] = df2["t"] - df2["t"].min()

        # Set all parent_id values to -1 for the first time point
        df2.loc[df2["t"] == 0, "parent_id"] = -1

        # find the tracks with parents at the first time point
        tracks_with_parents = df2.loc[
            (df2["t"] == 0) & (df2["parent_track_id"] != -1), "track_id"
        ]
        track_ids_to_update = set(tracks_with_parents)

        # update the parent_track_id to -1 for the tracks with parents at the first time point
        df2.loc[df2["track_id"].isin(track_ids_to_update), "parent_track_id"] = -1
        return df2

    def find_all_red_flags(self) -> pd.DataFrame:
        """
        Identify tracking red flags using multiple heuristics.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 't', 'track_id', 'id', 'event'
        """
        df = self.df_full.copy()

        # Apply different red flag detection heuristics
        rfs_starts_and_ends = find_all_starts_and_ends(df)
        self.overlapping_cells_df, rfs_overlap = find_overlapping_cells(
            df, self.db_path_new
        )

        # Trajectory changes (jumps and direction changes)
        # rfs_trajectory = find_trajectory_changes(df, self.scale)

        # Area/volume changes
        # rfs_area = find_area_changes(df)

        # Combine all red flag detection results
        # result_df = combine_red_flags(
        # rfs_starts_and_ends, rfs_overlap, rfs_trajectory, rfs_area
        # )

        result_df = combine_red_flags(rfs_starts_and_ends, rfs_overlap)

        # ToDo: make option to filter redflags in the first two timepoints
        # (useful for neuromast with suboptimal beginning)
        # result_df = result_df[result_df["t"] != 1]

        # Filter out red flags at the edge of the field of view
        if self.flag_remove_red_flags_at_edge:
            result_df = filter_red_flags_at_edge(
                red_flags=result_df,
                df_full=df,
                data_shape=self.data_shape_full[1:],
                edge_threshold=self.remove_red_flags_at_edge_threshold,
                ndim=self.ndim,
                scale=self.scale,
            )

        return result_df

    def find_all_divisions(self) -> pd.DataFrame:
        """
        Identify cell divisions by finding cells that have multiple daughters in the next timepoint.
        Returns only the last frame of each parent track before the division occurs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 't', 'track_id', 'id'
        """
        # Get all cells that have parents (parent_id != -1)
        cells_with_parents = self.df_full[self.df_full["parent_id"] != -1]

        # Group by parent_id and time to count daughters per parent per timepoint
        daughter_counts = cells_with_parents.groupby(["parent_id", "t"]).size()

        # Find cases where a parent has exactly 2 daughters
        dividing_cells = daughter_counts[daughter_counts == 2].reset_index()

        if dividing_cells.empty:
            return pd.DataFrame(columns=["t", "track_id", "id"])

        # Get the parent information for these division events
        divisions = []
        for _, row in dividing_cells.iterrows():
            # Find parent info in the frame before division
            parent_info = self.df_full[
                (self.df_full["id"] == row["parent_id"])
                & (self.df_full["t"] == row["t"] - 1)
            ]
            daughters_info = self.df_full[
                (self.df_full["parent_id"] == row["parent_id"])
                & (self.df_full["t"] == row["t"])
            ]
            if not parent_info.empty:
                divisions.append(
                    {
                        "t": parent_info["t"].iloc[0],
                        "track_id": parent_info["track_id"].iloc[0],
                        "id": row["parent_id"],
                        "daughters": daughters_info["track_id"].tolist(),
                    }
                )

        return pd.DataFrame(divisions)

    def find_all_toannotate(self) -> pd.DataFrame:
        """
        Find all unannotated segments within tracks (generic = -1), with their time ranges and IDs.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: track_id, first_t, last_t, first_id, last_id, sorted by first appearance time
        """
        # Get rows with no annotations
        unannotated = self.df_full[
            self.df_full["generic"] == NodeDB.generic.default.arg
        ]

        if unannotated.empty:
            return pd.DataFrame(
                columns=["track_id", "first_t", "last_t", "first_id", "last_id"]
            )

        # Group by track_id and find consecutive unannotated segments
        segments = []

        for track_id in unannotated["track_id"].unique():
            track_data = unannotated[unannotated["track_id"] == track_id].sort_values(
                "t"
            )

            if track_data.empty:
                continue

            # Find consecutive time segments
            current_start = track_data.iloc[0]["t"]
            current_start_id = track_data.iloc[0]["id"]
            prev_time = current_start

            for i in range(1, len(track_data)):
                current_time = track_data.iloc[i]["t"]
                current_id = track_data.iloc[i]["id"]

                # If there's a gap in time, save the current segment and start a new one
                if current_time != prev_time + 1:
                    segments.append(
                        {
                            "track_id": track_id,
                            "first_t": current_start,
                            "last_t": prev_time,
                            "first_id": current_start_id,
                            "last_id": track_data.iloc[i - 1]["id"],
                        }
                    )
                    current_start = current_time
                    current_start_id = current_id

                prev_time = current_time

            # Add the last segment
            segments.append(
                {
                    "track_id": track_id,
                    "first_t": current_start,
                    "last_t": prev_time,
                    "first_id": current_start_id,
                    "last_id": track_data.iloc[-1]["id"],
                }
            )

        # Convert to DataFrame and sort
        to_annotate = pd.DataFrame(segments)
        if not to_annotate.empty:
            # Ensure all numeric columns are integers
            to_annotate = to_annotate.astype(
                {
                    "track_id": int,
                    "first_t": int,
                    "last_t": int,
                    "first_id": int,
                    "last_id": int,
                }
            )
            to_annotate = to_annotate.sort_values(
                ["first_t", "last_t", "first_id"]
            ).reset_index(drop=True)

        return to_annotate

    def recompute_red_flags(self):
        """called by update_red_flags in TrackEditClass upon tracks_updated signal in TracksViewer"""
        self.red_flags = self.find_all_red_flags()

        # Only filter if we have any red flags
        if not self.red_flags.empty:
            self.red_flags = self.red_flags[
                ~self.red_flags[["id", "event"]]
                .apply(tuple, axis=1)
                .isin(self.red_flags_ignore_list[["id", "event"]].apply(tuple, axis=1))
            ]

    def recompute_divisions(self):
        """called by update_divisions in TrackEditClass upon tracks_updated signal in TracksViewer"""
        self.divisions = self.find_all_divisions()

    def recompute_toannotate(self):
        """called by update_toannotate in TrackEditClass upon tracks_updated signal in TracksViewer"""
        self.toannotate = self.find_all_toannotate()

    def _load_red_flags_ignore_list(self) -> pd.DataFrame:
        """Load the red flags ignore list from text file if it exists."""
        ignore_file_path = self.working_directory / "red_flags_ignore_list.txt"

        if ignore_file_path.exists():
            try:
                ignore_data = []
                with open(ignore_file_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith(
                            "#"
                        ):  # Skip empty lines and comments
                            try:
                                t, id_val, event = line.split(",")
                                ignore_data.append(
                                    {
                                        "t": int(t.strip()),
                                        "id": int(id_val.strip()),
                                        "event": event.strip(),
                                    }
                                )
                            except ValueError:
                                self.log(
                                    f"Warning: Skipping malformed line in ignore list: {line}"
                                )
                                continue

                ignore_list = pd.DataFrame(ignore_data)
                if not ignore_list.empty:
                    ignore_list = ignore_list.sort_values(["t", "id"]).reset_index(
                        drop=True
                    )
                self.log(
                    f"Loaded {len(ignore_list)} ignored red flags from {ignore_file_path}"
                )
                return ignore_list
            except Exception as e:
                self.log(f"Error loading ignore list: {e}. Starting with empty list.")
                return pd.DataFrame(columns=["t", "id", "event"])
        else:
            self.log("No ignore list file found. Starting with empty list.")
            return pd.DataFrame(columns=["t", "id", "event"])

    def _save_red_flags_ignore_list(self):
        """Save the red flags ignore list to text file."""
        ignore_file_path = self.working_directory / "red_flags_ignore_list.txt"
        try:
            with open(ignore_file_path, "w") as f:
                f.write("# Red flags ignore list - one entry per line: t, id, event\n")

                for _, row in self.red_flags_ignore_list.iterrows():
                    f.write(f"{row['t']}, {row['id']}, {row['event']}\n")

            self.log(
                f"Saved {len(self.red_flags_ignore_list)} ignored red flags to {ignore_file_path}"
            )
        except Exception as e:
            self.log(f"Error saving ignore list: {e}")

    def ignore_red_flag(self, id, event):
        # Get the time for the specific red flag
        time = self.red_flags.loc[
            (self.red_flags["id"] == id) & (self.red_flags["event"] == event), "t"
        ].values[0]

        # Add to ignore list DataFrame
        new_ignore_row = pd.DataFrame({"t": [time], "id": [id], "event": [event]})
        self.red_flags_ignore_list = pd.concat(
            [self.red_flags_ignore_list, new_ignore_row], ignore_index=True
        )

        # Sort by time then id to maintain consistent order
        self.red_flags_ignore_list = self.red_flags_ignore_list.sort_values(
            ["t", "id"]
        ).reset_index(drop=True)

        # Save to text file immediately
        self._save_red_flags_ignore_list()

        message = f"ignore red flag: cell {id} {event} " f"at time {time}"
        self.log(message)

        # remove the ignored red flag from the red flags
        # Use both id and event to ensure we only remove the specific red flag type
        self.red_flags = self.red_flags[
            ~self.red_flags[["id", "event"]]
            .apply(tuple, axis=1)
            .isin(self.red_flags_ignore_list[["id", "event"]].apply(tuple, axis=1))
        ]

    def export_tracks(self):
        """Export tracks to a CSV file"""
        print("exporting...")

        # tracks.csv
        csv_filename = self.working_directory / f"{self.extension_string}_tracks.csv"
        df_full_export = self.df_full.copy()
        df_full_export["annotation"] = df_full_export["generic"].map(
            self.annotation_mapping
        )
        try:
            df_full_export.to_csv(csv_filename, index=False)
        except Exception as e:
            from napari.utils.notifications import show_error

            show_error(f"Error exporting tracks.csv: {str(e)}")

        # annotations.csv
        csv_filename = (
            self.working_directory / f"{self.extension_string}_annotations.csv"
        )
        # Create one row per segment instead of per track
        annotations_df = self.df_full[["track_id", "t", "generic"]].copy()
        annotations_df["label"] = annotations_df["generic"].map(self.annotation_mapping)

        # Group by track_id and find segments with the same label
        segments = []
        for track_id in annotations_df["track_id"].unique():
            track_data = annotations_df[
                annotations_df["track_id"] == track_id
            ].sort_values("t")

            if track_data.empty:
                continue

            # Find consecutive frames with the same label
            current_label = track_data.iloc[0]["label"]
            current_start = track_data.iloc[0]["t"]
            prev_time = current_start

            for i in range(1, len(track_data)):
                current_time = track_data.iloc[i]["t"]
                current_label_at_time = track_data.iloc[i]["label"]

                # If there's a gap in time or label changes, save the current segment
                if (
                    current_time != prev_time + 1
                    or current_label_at_time != current_label
                ):
                    segments.append(
                        {
                            "track_id": track_id,
                            "first_t": int(current_start),
                            "last_t": int(prev_time),
                            "label": current_label,
                        }
                    )
                    current_start = current_time
                    current_label = current_label_at_time

                prev_time = current_time

            # Add the last segment
            segments.append(
                {
                    "track_id": track_id,
                    "first_t": int(current_start),
                    "last_t": int(prev_time),
                    "label": current_label,
                }
            )

        # Convert to DataFrame and export
        df_segments = pd.DataFrame(segments)
        if not df_segments.empty:
            df_segments = df_segments.sort_values(["track_id", "first_t"]).reset_index(
                drop=True
            )

        df_segments.to_csv(csv_filename, index=False)

        # divisions.txt
        txt_filename = self.working_directory / f"{self.extension_string}_divisions.txt"
        # Create a mapping of track_id to its label
        track_labels = (
            self.df_full.groupby("track_id")["generic"]
            .first()
            .apply(self.annotation_mapping)
        )

        with open(txt_filename, "w") as f:
            for _, row in self.divisions.iterrows():
                parent_id = row["track_id"]
                daughter_tracks = row["daughters"]
                if isinstance(
                    daughter_tracks, str
                ):  # Convert string representation to list if needed
                    daughter_tracks = eval(daughter_tracks)

                parent_label = track_labels.get(parent_id, "other")
                daughter1_label = track_labels.get(daughter_tracks[0], "other")
                daughter2_label = track_labels.get(daughter_tracks[1], "other")

                f.write(
                    f"division: {parent_id} ({parent_label}) > {daughter_tracks[0]} ({daughter1_label})"
                    f" + {daughter_tracks[1]} ({daughter2_label})\n"
                )

        # segments.zarr
        zarr_filename = (
            self.working_directory / f"{self.extension_string}_segments.zarr"
        )
        tracks_to_zarr(
            config=self.config_adjusted,
            tracks_df=self.df_full,
            store_or_path=zarr_filename,
            overwrite=True,
        )

        # annotations.zarr
        zarr_filename = (
            self.working_directory / f"{self.extension_string}_annotations.zarr"
        )
        annotations_to_zarr(
            config=self.config_adjusted,
            tracks_df=self.df_full,
            store_or_path=zarr_filename,
            overwrite=True,
        )
        print("exporting finished!")

    def annotation_mapping(self, label):
        """Map the label to the generic column."""
        return self.annotation_mapping_dict.get(label, {"name": "other"})["name"]

    @property
    def color_mapping(self):
        """Get the color mapping from the annotation_mapping_dict."""
        # Default gray color for unmapped values
        default_color = [0.5, 0.5, 0.5, 1.0]
        # Transparent color for 0 (background)
        transparent_color = [0.0, 0.0, 0.0, 0.0]

        # Create a dictionary that will store all possible integer values up to a reasonable maximum
        # Start with the explicit mappings from annotation_mapping_dict
        color_dict = {k: v["color"] for k, v in self.annotation_mapping_dict.items()}

        # Add special case for 0 to be transparent
        color_dict[0] = transparent_color
        color_dict[None] = transparent_color

        # Add mappings for all other possible values
        # Using a reasonable range that covers potential annotation values
        all_values = range(-1, 20)  # Adjust range as needed
        for val in all_values:
            if val not in color_dict:
                color_dict[val] = default_color

        return color_dict

    def annotate_track(
        self, track_id: int, label: int, t_begin: int = None, t_end: int = None
    ):
        """Annotate cells of a track in the database with a given label.

        Parameters
        ----------
        track_id : int
            The track ID to annotate
        label : int
            The annotation label to apply
        t_begin : int, optional
            Start time for partial annotation. If None, annotate from the beginning of the track.
        t_end : int, optional
            End time for partial annotation. If None, annotate to the end of the track.
        """

        # Find indices for the specified track and time range
        track_mask = self.df_full["track_id"] == track_id

        if t_begin is not None or t_end is not None:
            # Partial annotation: filter by time range
            time_mask = pd.Series(True, index=self.df_full.index)
            if t_begin is not None:
                time_mask &= self.df_full["t"] >= t_begin
            if t_end is not None:
                time_mask &= self.df_full["t"] <= t_end

            indices = self.df_full[track_mask & time_mask].index.tolist()
        else:
            # Full track annotation: annotate all cells of the track
            indices = self.df_full[track_mask].index.tolist()

        if indices:
            self.change_values(
                indices,
                NodeDB.generic,
                label,
                log_header="annotate_track:" + str(track_id),
            )

    def clear_nodes_annotations(self, nodes):
        """Clear the annotations for the entire track of a list of nodes. Called when a node is deleted."""

        print(f"clearing annotations for nodes: {nodes}")

        # get the track_id of the nodes
        track_ids = self.df_full[self.df_full["id"].isin(nodes)]["track_id"].unique()

        # Collect all node indices of which the track_id is in track_ids
        all_indices = []
        for track_id in track_ids:
            indices = self.df_full[self.df_full["track_id"] == track_id].index.tolist()
            all_indices.extend(indices)

        # Make a single call with all indices
        if all_indices:
            self.change_values(all_indices, NodeDB.generic, NodeDB.generic.default.arg)

    def clear_edges_annotations(self, edges):
        """Clear annotations for all nodes involved in the given edges.

        Parameters
        ----------
        edges : List[Tuple[int, int]]
            List of edges, where each edge is a tuple of (source_node, target_node)
        """

        print(f"clearing annotations for edges: {edges}")
        # Flatten the list of edge tuples to get all node IDs
        node_ids = set()
        for source, target in edges:
            node_ids.add(source)
            node_ids.add(target)

        # get the track_id of the nodes
        track_ids = self.df_full[self.df_full["id"].isin(node_ids)]["track_id"].unique()

        # Collect all indices that need to be changed
        all_indices = []
        for track_id in track_ids:
            indices = self.df_full[self.df_full["track_id"] == track_id].index.tolist()
            all_indices.extend(indices)

        # Only make the call if we have indices to change
        if all_indices:
            self.change_values(all_indices, NodeDB.generic, NodeDB.generic.default.arg)

    def check_zarr_existance(self):
        # check if zarr file and channel exists:
        if self.imaging_flag:
            if not self.imaging_zarr_file or not self.imaging_channel:
                self.imaging_flag = False
                print(
                    "Warning: Imaging zarr file or channel not specified. Imaging data not shown."
                )
                return

            # Check if the zarr root exists by looking for zarr.json (zarr v3) or .zgroup (zarr v2)
            zarr_root = Path(self.imaging_zarr_file)
            if (
                not (zarr_root / "zarr.json").exists()
                and not (zarr_root / ".zgroup").exists()
            ):
                self.imaging_flag = False
                print(
                    f"Warning: Not a valid zarr dataset at {self.imaging_zarr_file} (missing zarr.json or .zgroup). "
                    "Imaging data not shown."
                )
                return

            # Check if the channel is a valid zarr array by looking for zarr.json (zarr v3) or .zarray (zarr v2)
            channel_path = zarr_root / Path(self.imaging_channel)
            if (
                not (channel_path / "zarr.json").exists()
                and not (channel_path / ".zarray").exists()
            ):
                self.imaging_flag = False
                print(
                    f"Warning: Not a valid zarr array at channel path {self.imaging_channel} "
                    "(missing zarr.json or .zarray). "
                    "Imaging data not shown."
                )

    def check_annotation_mapping(self):
        """
        Check if the annotation mapping is valid by verifying:
        - All entries have required fields (name, color)
        - Name field is a string
        - Color field is a list of 4 numerical values
        - Names are unique across all entries
        - Values (labels) are unique
        - Values cannot be 0 (reserved values for background)

        Raises
        ------
        ValueError
            If any validation check fails
        """
        if not self.annotation_mapping_dict:
            raise ValueError("Annotation mapping dictionary is empty")

        # Keep track of names and values to check for duplicates
        used_names = set()
        used_values = set()

        for label, mapping in self.annotation_mapping_dict.items():
            # Check for reserved values (-1 and 0)
            if label == 0:
                raise ValueError(
                    f"Label value {label} is reserved and cannot be used in annotation mapping"
                )

            # Check required fields
            if not isinstance(mapping, dict):
                raise ValueError(f"Mapping for label {label} must be a dictionary")

            if not all(field in mapping for field in ["name", "color"]):
                raise ValueError(
                    f"Mapping for label {label} missing required fields 'name' and/or 'color'"
                )

            # Check name is string
            if not isinstance(mapping["name"], str):
                raise ValueError(
                    f"Name for label {label} must be a string, got {type(mapping['name'])}"
                )

            # Check for duplicate names
            if mapping["name"] in used_names:
                raise ValueError(
                    f"Duplicate name '{mapping['name']}' found in annotation mapping"
                )
            used_names.add(mapping["name"])

            # Check for duplicate values (labels)
            if label in used_values:
                raise ValueError(
                    f"Duplicate value '{label}' found in annotation mapping"
                )
            used_values.add(label)

            # Check color is list of 4 numbers
            if not isinstance(mapping["color"], (list, tuple)):
                raise ValueError(f"Color for label {label} must be a list or tuple")

            if len(mapping["color"]) != 4:
                raise ValueError(f"Color for label {label} must have exactly 4 values")

            if not all(isinstance(v, (int, float)) for v in mapping["color"]):
                raise ValueError(f"Color values for label {label} must all be numbers")

            if not all(0 <= v <= 1 for v in mapping["color"]):
                raise ValueError(
                    f"Color values for label {label} must be between 0 and 1"
                )

        # Keep the original functionality
        if self.annotation_mapping is None:
            self.annotation_mapping = {
                k: {"name": v["name"], "color": v["color"]}
                for k, v in self.annotation_mapping_dict.items()
            }
