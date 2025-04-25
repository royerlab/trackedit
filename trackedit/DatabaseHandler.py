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
from ultrack.core.export.utils import solution_dataframe_from_sql
from ultrack.tracks.graph import add_track_ids_to_tracks_df

from trackedit.arrays.DatabaseArray import DatabaseArray
from trackedit.arrays.ImagingArray import SimpleImageArray
from trackedit.utils.utils import annotations_to_zarr

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
        self.imaging_flag = True if self.imaging_zarr_file is not None else False

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
        if self.ndim == len(self.scale) + 1:
            self.z_scale = self.scale[0]
        else:
            self.z_scale = None
            raise ValueError(
                f"Expected scale with {self.ndim-1} values, (Z)YX, but scale has {len(self.scale)} values."
            )

        # change initial chunk depending on data shape
        if self.data_shape_full[0] < self.time_chunk_length:
            self.time_chunk_length = self.data_shape_full[0]
            self.time_chunk_overlap = 0

        # calculate time chunk
        self.time_window, self.time_chunk_starts = self.calc_time_window()
        self.data_shape_chunk = self.data_shape_full.copy()
        self.data_shape_chunk[0] = self.time_chunk_length
        self.num_time_chunks = len(
            np.arange(0, self.Tmax, self.time_chunk_length - self.time_chunk_overlap)
        )

        self.add_missing_columns_to_db()

        # DatabaseArray()
        self.segments = DatabaseArray(
            database_path=self.db_path_new,
            shape=self.data_shape_chunk,
            time_window=self.time_window,
            color_by_field=NodeDB.id,
        )
        self.annotArray = DatabaseArray(
            database_path=self.db_path_new,
            shape=self.data_shape_chunk,
            time_window=self.time_window,
            color_by_field=NodeDB.generic,
        )
        self.check_zarr_existance()
        if self.imaging_flag:
            self.imagingArray = SimpleImageArray(
                imaging_zarr_file=self.imaging_zarr_file,
                channel=self.imaging_channel,
                time_window=self.time_window,
            )
        self.df_full = self.db_to_df(entire_database=True)
        # ToDo: df_full might be very large for large datasets, but annotation/redflags/division need it
        self.nxgraph = self.df_to_nxgraph()
        self.red_flags = self.find_all_red_flags()
        self.toannotate = self.find_all_toannotate()
        self.divisions = self.find_all_divisions()
        self.red_flags_ignore_list = []
        self.log(f"Start annotation session ({datetime.now()})")

        # Default label for unlabeled cells
        default_annotation = {
            NodeDB.generic.default.arg: {  # -1
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

    def log(self, message):
        """Append a message to the log file."""
        with open(self.log_file, "a") as log:
            log.write(message + "\n")

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

        # Add the new generic column using the same default as NodeDB.generic
        generic_default = NodeDB.generic.default.arg if NodeDB.generic.default else None
        expected_columns[
            "generic"
        ] = f'INTEGER{" DEFAULT " + str(generic_default) if generic_default is not None else ""}'

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
        Generate the next version of a database filename.
        """
        name, ext = os.path.splitext(old_filename)
        old_filename = name + ext
        db_filename_new = old_filename
        log_filename_new = f"{name}_changelog.txt"
        return old_filename, db_filename_new, log_filename_new

    def change_values(self, indices, field, values):
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
        """
        # Convert single index to list
        if isinstance(indices, (int, np.integer)):
            indices = [int(indices)]
        else:
            indices = [int(i) for i in indices]

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
            self.log(message)

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
        time_window = (
            int(time_chunk_starts[self.time_chunk]),
            int(time_chunk_stops[self.time_chunk]),
        )
        return time_window, time_chunk_starts

    def set_time_chunk(self, time_chunk):
        if time_chunk >= self.num_time_chunks:
            raise ValueError(
                f"Time chunk {time_chunk} out of range. Maximum time chunk is {self.num_time_chunks-1}"
            )
        else:
            self.time_chunk = time_chunk
            self.time_window, _ = self.calc_time_window()
            self.segments.set_time_window(self.time_window)
            self.annotArray.set_time_window(self.time_window)
            if self.imaging_flag:
                self.imagingArray.set_time_window(self.time_window)
            self.nxgraph = self.df_to_nxgraph()

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

        df = solution_dataframe_from_sql(
            self.db_path_new,
            columns=(
                NodeDB.id,
                NodeDB.parent_id,
                NodeDB.t,
                NodeDB.z,
                NodeDB.y,
                NodeDB.x,
                NodeDB.generic,
            ),
        )
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
        if self.ndim == 4:
            df_scaled.loc[:, "z"] = df_scaled.z * self.z_scale  # apply scale

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
        Identify tracking red flags ('added' or 'removed') from one timepoint to the next.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: 't', 'track_id', 'id', 'event'
        """
        df = self.df_full.copy()

        # Define a continuous range of timepoints.
        time_range = range(df["t"].min(), df["t"].max() + 1)

        # Precompute the set of track_ids for each timepoint.
        track_sets = (
            df.groupby("t")["track_id"].agg(set).reindex(time_range, fill_value=set())
        )
        # Precompute the set of cell ids for each timepoint.
        id_sets = df.groupby("t")["id"].agg(set).reindex(time_range, fill_value=set())

        # Precompute mappings from (t, track_id) to the cell's own id and parent_id.
        id_mapping = df.set_index(["t", "track_id"])["id"].to_dict()
        parent_mapping = df.set_index(["t", "track_id"])["parent_id"].to_dict()

        # For each timepoint, count how many times a given id appears as a parent_id (i.e. number of daughters).
        daughter_counts = {t: {} for t in time_range}
        for t, group in df.groupby("t"):
            daughter_counts[t] = group["parent_id"].value_counts().to_dict()

        events = []
        timepoints = list(time_range)

        # Before the main loop, let's identify single-point tracks
        track_lengths = df.groupby("track_id").size()
        single_point_tracks = set(track_lengths[track_lengths == 1].index)

        for i, t in enumerate(timepoints):
            current_tracks = track_sets[t]
            # For t=0, use the current set as the "previous" set.
            prev_tracks = track_sets[timepoints[i - 1]] if i > 0 else current_tracks
            # For the last timepoint, use the current set as the "next" set.
            next_tracks = (
                track_sets[timepoints[i + 1]]
                if i < len(timepoints) - 1
                else current_tracks
            )

            # Detect "added" events: cells present now but not in the previous timepoint.
            added_tracks = current_tracks - prev_tracks
            for track in added_tracks:
                # Check if this row has a valid parent.
                par = parent_mapping.get((t, track), -1)
                # If the cell has a parent (par != -1) and that parent exists in the previous timepoint,
                # then it is likely due to a division. In that case, skip flagging it.
                if par != -1 and (i > 0 and par in id_sets[timepoints[i - 1]]):
                    continue
                events.append(
                    {
                        "t": t,
                        "track_id": track,
                        "id": id_mapping.get((t, track)),
                        "event": "added",
                    }
                )

            # Detect "removed" events: cells present now but not in the next timepoint.
            removed_tracks = current_tracks - next_tracks
            for track in removed_tracks:
                cell_id = id_mapping.get((t, track))
                # Skip if this is a single-point track (it will be reported as 'added' only)
                if track in single_point_tracks:
                    continue
                # Check for division: in the next timepoint, if there are 2 or more cells
                # with parent_id equal to this cell's id, skip flagging.
                if i < len(timepoints) - 1:
                    daughters = daughter_counts.get(timepoints[i + 1], {})
                    if daughters.get(cell_id, 0) >= 2:
                        continue
                events.append(
                    {
                        "t": t,
                        "track_id": track,
                        "id": cell_id,
                        "event": "removed",
                    }
                )

        result_df = pd.DataFrame(events)

        # If no events were found, return empty DataFrame with correct columns
        if result_df.empty:
            return pd.DataFrame(columns=["t", "track_id", "id", "event"])

        # ToDo: make option to filter redflags in the first two timepoints
        # (useful for neuromast with suboptimal beginning)
        # result_df = result_df[result_df["t"] != 1]

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
        Find all track IDs that have no annotations (generic = 0), with their mean appearance time and first ID.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: track_id, first_t, first_id, sorted by mean appearance time
        """
        # Get rows with no annotations
        unannotated = self.df_full[
            self.df_full["generic"] == NodeDB.generic.default.arg
        ]

        # For each track_id, get the first time point and corresponding first ID
        to_annotate = (
            unannotated.groupby("track_id")
            .agg({"t": "first"})  # Get the first time point of each track
            .astype({"t": int})  # Convert first time to integer
            .reset_index()
            .rename(columns={"t": "first_t"})
            .sort_values("first_t")
            .reset_index(drop=True)
        )

        # Get the IDs at these first times
        to_annotate = to_annotate.merge(
            unannotated[["track_id", "t", "id"]],
            left_on=["track_id", "first_t"],
            right_on=["track_id", "t"],
        )[["track_id", "first_t", "id"]].rename(columns={"id": "first_id"})

        return to_annotate

    def recompute_red_flags(self):
        """called by update_red_flags in TrackEditClass upon tracks_updated signal in TracksViewer"""
        self.red_flags = self.find_all_red_flags()

        # Only filter if we have any red flags
        if not self.red_flags.empty:
            self.red_flags = self.red_flags[
                ~self.red_flags["id"].isin(self.red_flags_ignore_list)
            ]

    def recompute_divisions(self):
        """called by update_divisions in TrackEditClass upon tracks_updated signal in TracksViewer"""
        self.divisions = self.find_all_divisions()

    def recompute_toannotate(self):
        """called by update_toannotate in TrackEditClass upon tracks_updated signal in TracksViewer"""
        self.toannotate = self.find_all_toannotate()

    def ignore_red_flag(self, id):
        self.red_flags_ignore_list.append(id)

        # Get the values first for clarity
        event = self.red_flags.loc[self.red_flags["id"] == id, "event"].values[0]
        time = self.red_flags.loc[self.red_flags["id"] == id, "t"].values[0]

        message = f"ignore red flag: cell {id} {event} " f"at time {time}"
        self.log(message)

        # remove the ignores red flag from the red flags
        self.red_flags = self.red_flags[
            ~self.red_flags["id"].isin(self.red_flags_ignore_list)
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
        # Group by track_id and take the first label for each track
        annotations_df = self.df_full[["track_id", "t", "generic"]].copy()
        annotations_df["label"] = annotations_df["generic"].map(self.annotation_mapping)
        df_grouped = annotations_df.groupby("track_id").first().reset_index()
        df_grouped.to_csv(csv_filename, index=False)
        # ToDo: check if only one label per track_id

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

    def annotate_track(self, track_id: int, label: int):
        """Annotate all cells of a track in the database with a given label."""

        # Then find this track_id in the toannotate
        indices = self.df_full[self.df_full["track_id"] == track_id].index.tolist()

        self.change_values(indices, NodeDB.generic, label)

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

            # Check if the zarr root exists by looking for .zgroup
            zarr_root = Path(self.imaging_zarr_file)
            if not (zarr_root / ".zgroup").exists():
                self.imaging_flag = False
                print(
                    f"Warning: Not a valid zarr dataset at {self.imaging_zarr_file} (missing .zgroup). "
                    "Imaging data not shown."
                )
                return

            # Check if the channel is a valid zarr array by looking for .zarray
            channel_path = zarr_root / Path(self.imaging_channel)
            if not (channel_path / ".zarray").exists():
                self.imaging_flag = False
                print(
                    f"Warning: Not a valid zarr array at channel path {self.imaging_channel} (missing .zarray). "
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
