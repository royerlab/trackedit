import pandas as pd
import sqlalchemy as sqla
from ultrack.core.database import OverlapDB, Session


def find_all_starts_and_ends(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify tracking red flags ('added' or 'removed') from one timepoint to the next.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: track_id, t, z, y, x, parent_id, id

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 't', 'track_id', 'id', 'event'
    """
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
            track_sets[timepoints[i + 1]] if i < len(timepoints) - 1 else current_tracks
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
                    "id": cell_id,
                    "event": "removed",
                }
            )

    result_df = pd.DataFrame(events)

    # If no events were found, return empty DataFrame with correct columns
    if result_df.empty:
        return pd.DataFrame(columns=["t", "id", "event"])

    return result_df


def find_overlapping_cells(df_full: pd.DataFrame, database_path: str) -> pd.DataFrame:
    """
    Find red flags based on overlapping cells heuristic.

    Parameters
    ----------
    df_full : pd.DataFrame
        DataFrame with cell information indexed by cell id
    database_path : str
        Path to the database containing OverlapDB table

    Returns
    -------
    pd.DataFrame
        DataFrame with overlapping cell pairs where both cells exist in df_full
    """
    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        overlap_query = session.query(
            OverlapDB.node_id,
            OverlapDB.ancestor_id,
        )

        overlap_df = pd.read_sql(
            overlap_query.statement,
            session.bind,
        )

    # Filter overlap_df to only include rows where both node_id and ancestor_id
    # exist in the index of df_full
    if overlap_df.empty:
        return pd.DataFrame(columns=["node_id", "ancestor_id"]), pd.DataFrame(
            columns=["t", "id", "event"]
        )

    # Get the set of valid cell ids from df_full index
    valid_ids = set(df_full.index)

    # Filter overlap_df to only include rows where both ids exist
    filtered_overlap_df = overlap_df[
        (overlap_df["node_id"].isin(valid_ids))
        & (overlap_df["ancestor_id"].isin(valid_ids))
    ]

    # Create red flag format DataFrame
    red_flag_events = []
    for _, row in filtered_overlap_df.iterrows():
        # Get time information for both cells from df_full
        node_time = (
            df_full.loc[row["node_id"], "t"]
            if row["node_id"] in df_full.index
            else None
        )
        ancestor_time = (
            df_full.loc[row["ancestor_id"], "t"]
            if row["ancestor_id"] in df_full.index
            else None
        )

        # Add both cells as overlap events
        if node_time is not None:
            red_flag_events.append(
                {"t": node_time, "id": row["node_id"], "event": "overlap"}
            )

        if ancestor_time is not None:
            red_flag_events.append(
                {"t": ancestor_time, "id": row["ancestor_id"], "event": "overlap"}
            )

    red_flag_df = pd.DataFrame(red_flag_events)

    return filtered_overlap_df, red_flag_df


def combine_red_flags(*red_flag_dfs: pd.DataFrame) -> pd.DataFrame:
    """w
    Combine multiple red flag DataFrames into a single DataFrame.

    Parameters
    ----------
    *red_flag_dfs : pd.DataFrame
        Multiple DataFrames with columns: 't', 'track_id', 'id', 'event'

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all red flags, sorted by time
    """
    if not red_flag_dfs:
        return pd.DataFrame(columns=["t", "track_id", "id", "event"])

    # Combine all DataFrames
    combined_df = pd.concat(red_flag_dfs, ignore_index=True)

    # Remove duplicates based on (t, id, event) combination
    combined_df = combined_df.drop_duplicates(subset=["t", "id", "event"])

    # Sort by time then id for consistent ordering
    if not combined_df.empty:
        combined_df = combined_df.sort_values(["t", "id"]).reset_index(drop=True)

    return combined_df


def filter_red_flags_at_edge(
    red_flags: pd.DataFrame,
    df_full: pd.DataFrame,
    data_shape: tuple,
    edge_threshold: int,
    ndim: int,
) -> pd.DataFrame:
    """
    Filter out 'added' and 'removed' red flags near the edge of the field of view (FOV).

    Red flags with event type 'added' or 'removed' at the edge are often false
    positives caused by cells entering or leaving the imaging volume. Overlap
    events are kept regardless of position since they represent real spatial
    conflicts between cells.

    Parameters
    ----------
    red_flags : pd.DataFrame
        DataFrame with red flag information, must have 'id' and 'event' columns
    df_full : pd.DataFrame
        DataFrame with all cell data, must have 'id', 'z', 'y', 'x' columns
        (indexed by 'id' or with 'id' as a column)
    data_shape : tuple
        Spatial dimensions of the data (excluding time):
        - For 3D data: (z_max, y_max, x_max)
        - For 2D data: (y_max, x_max)
    edge_threshold : int
        Distance threshold in pixels - cells within this distance from any
        edge are considered "at edge"
    ndim : int
        Number of dimensions: 4 for 3D+time, 3 for 2D+time

    Returns
    -------
    pd.DataFrame
        Filtered red flags DataFrame with edge-related 'added'/'removed' events removed,
        but all 'overlap' events preserved
    """
    if red_flags.empty:
        return red_flags

    # Separate overlap events (always keep) from added/removed events (filter at edge)
    overlap_mask = red_flags["event"] == "overlap"
    overlap_red_flags = red_flags[overlap_mask]
    non_overlap_red_flags = red_flags[~overlap_mask]

    # If no non-overlap red flags, return all overlaps
    if non_overlap_red_flags.empty:
        return overlap_red_flags.reset_index(drop=True)

    # Ensure df_full has 'id' as a column for merging (not as index)
    # Handle case where 'id' might be index, column, or both
    if df_full.index.name == "id":
        if "id" in df_full.columns:
            # 'id' is both index and column - just drop the index, keep the column
            df_full = df_full.reset_index(drop=True)
        else:
            # 'id' is only the index - convert it to a column
            df_full = df_full.reset_index(drop=False)
    elif "id" not in df_full.columns:
        # If 'id' is neither index nor column, raise error
        raise ValueError(
            "df_full must have 'id' as either index or column for red flag filtering"
        )

    # Merge non-overlap red flags with cell position data
    red_flags_with_pos = non_overlap_red_flags.merge(
        df_full[["id", "z", "y", "x"]], on="id", how="left"
    )

    # Calculate distance to edges for each red flag
    if ndim == 4:
        # 3D data: check z, y, x
        z_max, y_max, x_max = data_shape

        distances_to_edges = pd.DataFrame(
            {
                "z_min": red_flags_with_pos["z"],
                "z_max": z_max - red_flags_with_pos["z"] - 1,
                "y_min": red_flags_with_pos["y"],
                "y_max": y_max - red_flags_with_pos["y"] - 1,
                "x_min": red_flags_with_pos["x"],
                "x_max": x_max - red_flags_with_pos["x"] - 1,
            }
        )
    else:
        # 2D data: check y, x only
        y_max, x_max = data_shape

        distances_to_edges = pd.DataFrame(
            {
                "y_min": red_flags_with_pos["y"],
                "y_max": y_max - red_flags_with_pos["y"] - 1,
                "x_min": red_flags_with_pos["x"],
                "x_max": x_max - red_flags_with_pos["x"] - 1,
            }
        )

    # Find minimum distance to any edge for each red flag
    min_distance_to_edge = distances_to_edges.min(axis=1)

    # Keep only 'added'/'removed' red flags that are NOT at the edge
    at_edge_mask = min_distance_to_edge <= edge_threshold
    filtered_non_overlap = non_overlap_red_flags[~at_edge_mask]

    # Combine filtered non-overlap events with all overlap events
    filtered_red_flags = pd.concat(
        [filtered_non_overlap, overlap_red_flags], ignore_index=True
    )

    # Sort by time and id for consistent ordering
    if not filtered_red_flags.empty:
        filtered_red_flags = filtered_red_flags.sort_values(["t", "id"]).reset_index(
            drop=True
        )

    return filtered_red_flags
