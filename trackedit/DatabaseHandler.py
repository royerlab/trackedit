import re
import os
import shutil
import networkx as nx
from pathlib import Path
from typing import List

from ultrack.config import MainConfig
from ultrack.core.database import *
from ultrack.core.export.utils import solution_dataframe_from_sql
from ultrack.tracks.graph import add_track_ids_to_tracks_df
from ultrack.core.export import tracks_layer_to_networkx
from motile_toolbox.candidate_graph import NodeAttr

from trackedit.DatabaseArray import DatabaseArray

class DatabaseHandler():
    def __init__(self, 
                 db_filename_old: str,
                 working_directory: Path, 
                 data_shape_full: List[int],        #T(Z)YX
                 z_scale: tuple,
                 name: str,
                 time_chunk: int = 0,
                 time_chunk_length: int = 105,
                 time_chunk_overlap: int = 5,
                 allow_overwrite: bool =False):

        #inputs
        self.db_filename_old = db_filename_old
        self.working_directory = working_directory
        self.data_shape_full = data_shape_full
        self.z_scale = z_scale
        self.name = name
        self.allow_overwrite = allow_overwrite
        self.time_chunk = time_chunk
        self.time_chunk_length = time_chunk_length
        self.time_chunk_overlap = time_chunk_overlap

        #calculate time chunk
        self.time_window = self.calc_time_window()
        self.data_shape_chunk = self.data_shape_full.copy()
        self.data_shape_chunk[0] = time_chunk_length
        self.num_time_chunks = int(np.ceil(self.data_shape_full[0]/self.time_chunk_length))

        #Filenames / directories
        self.db_filename_new, self.log_filename_new = self.copy_database(
                        self.working_directory, 
                        self.db_filename_old,
                        allow_overwrite=self.allow_overwrite)
        self.db_path_new = f"sqlite:///{self.working_directory/self.db_filename_new}"
        self.log_file = self.initialize_logfile(self.log_filename_new)

        self.config_adjusted = self.initialize_config()

        #DatabaseArray()
        self.segments = DatabaseArray(database_path=self.db_path_new, 
                                 shape=self.data_shape_chunk,
                                 time_window = self.time_window)
        self.df = self.db_to_df()
        self.nxgraph = self.df_to_nxgraph()
        self.log(f"Log file created")


    def initialize_logfile(self, log_filename_new):
        """Initialize the logger with a file path. Raises an error if the file already exists."""

        log_file_path = self.working_directory/log_filename_new

        if os.path.exists(log_file_path):
            if not self.allow_overwrite:
                raise FileExistsError(f"Log file '{log_file_path}' already exists. Choose a different file or delete the existing one.")
            else:
                open(log_file_path, "w").close()  # Clear the file if it exists

        print('old database:',self.db_filename_old)
        print('new database:',self.db_filename_new)
        print('new logfile:',log_filename_new)

        return log_file_path


    def log(self, message):
        """Append a message to the log file."""
        with open(self.log_file, "a") as log:
            log.write(message + "\n")

    def initialize_config(self):
        #import db filename properly into an Ultrack config, neccesary for chaning values in database
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
            raise FileNotFoundError(f"Error: {db_filename_old} not found in the working directory.")
        
        # Check if the new database file already exists
        if new_db_path.exists() & (not allow_overwrite):
            raise FileExistsError(f"Error: {db_filename_new} already exists. Copy operation aborted.")
        else:
            shutil.copy(old_db_path, new_db_path)
            print(f"Database copied to: {new_db_path}")

        return db_filename_new, log_filename_new
    
    def get_next_db_filename(self,old_filename):
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
        db_filename_new = f"{base_name}_v{version_number}{ext}"
        log_filename_new = f"data_v{version_number}_changelog.txt"
        return db_filename_new, log_filename_new
    
    def change_value(self, index, field, value):
        index = [int(index)]
        value = [int(value)]

        old_val = get_node_values(self.config_adjusted.data_config,
                    indices=index,
                    values=field)
        set_node_values(self.config_adjusted.data_config,
                    indices=index,
                    **{field.name: value})
        new_val = get_node_values(self.config_adjusted.data_config,
                    indices=index,
                    values=field)
        message = f"db: setting {field.name}[id={index}] = {new_val} (was {old_val})"
        # print(' '+message)
        self.log(message)

    def calc_time_window(self):
        Tmax = self.data_shape_full[0]
        time_chunk_starts = np.arange(0,Tmax,self.time_chunk_length-self.time_chunk_overlap)
        time_chunk_stops = np.array([(s + self.time_chunk_length) if (s + self.time_chunk_length < Tmax) else Tmax for s in time_chunk_starts]).astype(int)
        time_window = (int(time_chunk_starts[self.time_chunk]),int(time_chunk_stops[self.time_chunk]))
        return time_window
    

    def set_time_chunk(self, time_chunk):
        #ToDo: separate into two functions, one for setting time chunk and one for updating the graph/segments > maybe not necessary!
        if time_chunk >= self.num_time_chunks:
            raise ValueError(f"Time chunk {time_chunk} out of range. Maximum time chunk is {self.num_time_chunks-1}")
        else:
            self.time_chunk = time_chunk
            self.time_window = self.calc_time_window()
            self.segments.set_time_window(self.time_window)
            self.df = self.db_to_df()
            self.nxgraph = self.df_to_nxgraph()

    def db_to_df(self,       
                entire_database: bool = False, 
                include_parents: bool = True,
                include_node_ids: bool = True
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

        df = self.remove_past_parents_from_df(df)

        if ndim == 4:
            df.loc[:,"z"] = df.z * self.z_scale   #apply scale
            columns = ["track_id", "t", "z", "y", "x"]
        elif ndim == 3:
            columns = ["track_id", "t", "y", "x"]
        else:
            raise ValueError(
                f"Expected dataset with 3 or 4 dimensions, T(Z)YX. Found {ndim}."
            )
        if include_node_ids:
            df.loc[:,"id"] = df.index
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
        nxgraph = tracks_layer_to_networkx(self.df)

        # add/modify attributes to fit motile viewer assumptions
        for node in nxgraph.nodes:
            nxgraph.nodes[node][NodeAttr.TRACK_ID.value] = int(nxgraph.nodes[node]['track_id'])

        return nxgraph        

    def remove_past_parents_from_df(self, df2):

        df2.loc[:,"t"] = df2["t"] - df2["t"].min()

        # Set all parent_id values to -1 for the first time point
        df2.loc[df2["t"] == 0, "parent_id"] = -1

        #find the tracks with parents at the first time point
        tracks_with_parents = df2.loc[(df2["t"] == 0) & (df2["parent_track_id"] != -1), "track_id"]
        track_ids_to_update = set(tracks_with_parents)

        # update the parent_track_id to -1 for the tracks with parents at the first time point
        df2.loc[df2["track_id"].isin(track_ids_to_update), "parent_track_id"] = -1
        return df2
