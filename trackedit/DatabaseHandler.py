import os
import re
import shutil
from pathlib import Path
from typing import List

import networkx as nx
from motile_toolbox.candidate_graph import NodeAttr
from sqlalchemy import create_engine, inspect, text
from ultrack.config import MainConfig
from ultrack.core.database import *
from ultrack.core.export import tracks_layer_to_networkx, tracks_to_zarr
from ultrack.core.export.utils import solution_dataframe_from_sql
from ultrack.tracks.graph import add_track_ids_to_tracks_df

from trackedit.arrays.DatabaseArray import DatabaseArray


class DatabaseHandler:
    def __init__(
        self,
        db_filename_old: str,
        working_directory: Path,
        data_shape_full: List[int],  # T(Z)YX
        scale: tuple,
        name: str,
        time_chunk: int = 0,
        time_chunk_length: int = 105,
        time_chunk_overlap: int = 5,
        allow_overwrite: bool = False,
    ):

        # inputs
        self.db_filename_old = db_filename_old
        self.working_directory = working_directory
        self.data_shape_full = data_shape_full
        self.Tmax = self.data_shape_full[0]
        self.scale = scale
        self.z_scale = self.scale[0]
        self.name = name
        self.allow_overwrite = allow_overwrite
        self.time_chunk = time_chunk
        self.time_chunk_length = time_chunk_length
        self.time_chunk_overlap = time_chunk_overlap

        # calculate time chunk
        self.time_window, self.time_chunk_starts = self.calc_time_window()
        self.data_shape_chunk = self.data_shape_full.copy()
        self.data_shape_chunk[0] = time_chunk_length
        self.num_time_chunks = int(
            np.ceil(self.data_shape_full[0] / self.time_chunk_length)
        )

        # Filenames / directories
        self.extension_string = ""
        self.db_filename_new, self.log_filename_new = self.copy_database(
            self.working_directory,
            self.db_filename_old,
            allow_overwrite=self.allow_overwrite,
        )
        self.db_path_new = f"sqlite:///{self.working_directory/self.db_filename_new}"
        self.log_file = self.initialize_logfile(self.log_filename_new)

        self.config_adjusted = self.initialize_config()
        self.config_adjusted.data_config.metadata_add({"shape": self.data_shape_full})
        self.add_missing_columns_to_db()

        # DatabaseArray()
        self.segments = DatabaseArray(
            database_path=self.db_path_new,
            shape=self.data_shape_chunk,
            time_window=self.time_window,
        )
        self.df = self.db_to_df()
        self.nxgraph = self.df_to_nxgraph()
        self.red_flags = self.find_all_red_flags()
        self.divisions = self.find_all_divisions()
        self.red_flags_ignore_list = []
        self.log(f"Log file created")

    def initialize_logfile(self, log_filename_new):
        """Initialize the logger with a file path. Raises an error if the file already exists."""

        log_file_path = self.working_directory / log_filename_new

        if os.path.exists(log_file_path):
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

    def copy_database(self, working_directory, db_filename_old, allow_overwrite=False):
        """
        Copy the database to a new versioned filename.
        Ensures the new filename does not already exist before copying.
        """
        # Determine the new database filename
        db_filename_new, log_filename_new = self.get_next_db_filename(db_filename_old)

        # Create full paths
        old_db_path = Path(working_directory) / db_filename_old
        new_db_path = Path(working_directory) / db_filename_new

        # Check if the old database file exists
        if not old_db_path.exists():
            raise FileNotFoundError(
                f"Error: {db_filename_old} not found in the working directory."
            )

        # Check if the new database file already exists
        if new_db_path.exists() & (not allow_overwrite):
            raise FileExistsError(
                f"Error: {db_filename_new} already exists. Copy operation aborted."
            )
        else:
            shutil.copy(old_db_path, new_db_path)
            print(f"Database copied to: {new_db_path}")

        return db_filename_new, log_filename_new

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
        return expected_columns

    def get_next_db_filename(self, old_filename):
        """
        Generate the next version of a database filename.

        - If the filename is 'data.db', it returns 'data_v1.db'.
        - If the filename is 'data_v3.db', it returns 'data_v4.db'.
        """
        # Separate the file name and extension
        name, ext = os.path.splitext(old_filename)

        # Check if the filename already has a version (e.g., 'data_v3')
        match = re.match(r"^(.*)_v(\d+)$", name)

        if match:
            base_name = match.group(1)  # Extract name before "_v"
            version_number = int(match.group(2)) + 1  # Increment version
        else:
            base_name = name  # No versioning found, use the original name
            version_number = 1  # Start with version 1

        # Create the new filename
        self.extension_string = (
            f"_v{version_number}"  # "_v3" (saved for export function)
        )
        db_filename_new = f"{base_name}{self.extension_string}{ext}"
        log_filename_new = f"data_v{version_number}_changelog.txt"
        return db_filename_new, log_filename_new

    def change_value(self, index, field, value):
        index = [int(index)]
        value = [int(value)]

        old_val = get_node_values(
            self.config_adjusted.data_config, indices=index, values=field
        )
        set_node_values(
            self.config_adjusted.data_config, indices=index, **{field.name: value}
        )
        new_val = get_node_values(
            self.config_adjusted.data_config, indices=index, values=field
        )
        message = f"db: setting {field.name}[id={index}] = {new_val} (was {old_val})"
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
        # ToDo: separate into two functions, one for setting time chunk and one for updating the graph/segments > maybe not necessary!
        if time_chunk >= self.num_time_chunks:
            raise ValueError(
                f"Time chunk {time_chunk} out of range. Maximum time chunk is {self.num_time_chunks-1}"
            )
        else:
            self.time_chunk = time_chunk
            self.time_window, _ = self.calc_time_window()
            self.segments.set_time_window(self.time_window)
            self.df = self.db_to_df()
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
        ndim = len(self.data_shape_full)

        df = solution_dataframe_from_sql(self.db_path_new)
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

        if ndim == 4:
            columns = ["track_id", "t", "z", "y", "x"]
        elif ndim == 3:
            columns = ["track_id", "t", "y", "x"]
        else:
            raise ValueError(
                f"Expected dataset with 3 or 4 dimensions, T(Z)YX. Found {ndim}."
            )
        if include_node_ids:
            df.loc[:, "id"] = df.index
            columns.append("id")

        if include_parents:
            columns.append("parent_track_id")

            if include_node_ids:
                columns.append("parent_id")

        df = df[columns]
        return df

    def df_to_nxgraph(self) -> nx.DiGraph:
        """Exports solution from tracks dataframe to networkx graph.

        Returns
        nx.DiGraph
            Networkx graph.

        """
        # apply scale, only do this here to avoid scaling the original dataframe
        df_scaled = self.df.copy()
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
        For single-point tracks (tracks that exist at only one timepoint), only the 'added'
        event is reported to avoid duplicate flags for a single cell.
        """
        df = self.db_to_df(entire_database=True)

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

        # ignore events at t=1
        result_df = result_df[result_df["t"] != 1]

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
        df = self.db_to_df(entire_database=True)

        # Get all cells that have parents (parent_id != -1)
        cells_with_parents = df[df["parent_id"] != -1]

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
            parent_info = df[(df["id"] == row["parent_id"]) & (df["t"] == row["t"] - 1)]

            if not parent_info.empty:
                divisions.append(
                    {
                        "t": parent_info["t"].iloc[0],
                        "track_id": parent_info["track_id"].iloc[0],
                        "id": row["parent_id"],
                    }
                )

        return pd.DataFrame(divisions)

    def recompute_red_flags(self):
        """called by update_red_flags in TrackEditClass upon tracks_updated signal in TracksViewer"""
        self.red_flags = self.find_all_red_flags()
        self.red_flags = self.red_flags[
            ~self.red_flags["id"].isin(self.red_flags_ignore_list)
        ]

    def recompute_divisions(self):
        """called by update_divisions in TrackEditClass upon tracks_updated signal in TracksViewer"""
        self.divisions = self.find_all_divisions()

    def seg_ignore_red_flag(self, id):
        self.red_flags_ignore_list.append(id)

        message = f"ignore red flag: cell {id} {self.red_flags.loc[self.red_flags['id'] == id, 'event'].values[0]} at time {self.red_flags.loc[self.red_flags['id'] == id, 't'].values[0]} "
        self.log(message)

        # remove the ignores red flag from the red flags
        self.red_flags = self.red_flags[
            ~self.red_flags["id"].isin(self.red_flags_ignore_list)
        ]

    def export_tracks(self):
        """Export tracks to a CSV file"""
        print("exporting...")

        # CSV
        csv_filename = self.working_directory / f"tracks{self.extension_string}.csv"
        df_full = self.db_to_df(entire_database=True)
        try:
            df_full.to_csv(csv_filename, index=False)
        except Exception as e:
            from napari.utils.notifications import show_error

            show_error(f"Error exporting tracks.csv: {str(e)}")

        # Zarr
        zarr_filename = self.working_directory / f"segments{self.extension_string}.zarr"
        tracks_to_zarr(
            config=self.config_adjusted,
            tracks_df=df_full,
            store_or_path=zarr_filename,
            overwrite=True,
        )

        print("exporting finished!")
