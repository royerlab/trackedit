from pathlib import Path

import click
import geff
import numpy as np
import sqlalchemy as sa
from sqlalchemy.orm import Session
from tqdm import tqdm
from ultrack.core.database import Base, LinkDB, NodeDB
from ultrack.core.segmentation.node import Node
from ultrack.core.segmentation.processing import _generate_id

_max_segments_per_time = 1_000_000


def convert_geff_to_db(geff_path: Path, output_path: Path = None) -> None:
    """Convert GEFF file to ULTrack SQLite database.

    Args:
        geff_path: Path to the input GEFF file
        output_path: Optional path to the output database file.
                    If None, defaults to <input_stem>_to_db.db
    """
    # Determine output database path
    if output_path is None:
        database_path = geff_path.parent / f"{geff_path.stem}_to_db.db"
    else:
        database_path = output_path

    # Remove existing database if it exists
    if database_path.exists():
        database_path.unlink()
        print(f"Removed existing database at {database_path}")

    # Create engine and tables
    engine = sa.create_engine(f"sqlite:///{database_path}")
    Base.metadata.create_all(engine)
    print(f"Created new database at {database_path}")

    # Read GEFF file
    print("Reading GEFF file...")
    rx_graph, geff_metadata = geff.read(str(geff_path), backend="rustworkx")

    # Validate GEFF metadata
    required_node_props = ["t", "z", "y", "x", "solution", "mask", "bbox"]
    node_props = geff_metadata.node_props_metadata
    missing_props = [prop for prop in required_node_props if prop not in node_props]
    if missing_props:
        raise ValueError(f"Missing required node properties in GEFF: {missing_props}")

    print(f"Found {rx_graph.num_nodes()} nodes and {rx_graph.num_edges()} edges")

    # Get node ID mapping from graph attributes
    to_rx_id_map = rx_graph.attrs.get("to_rx_id_map", {})
    rx_to_original_id = {v: k + 1 for k, v in to_rx_id_map.items()}

    # First pass: determine if data is 2D or 3D by checking z-coordinate variation
    z_coords = []
    for rx_node_id in rx_graph.node_indices():
        node_attrs = rx_graph[rx_node_id]
        z_coords.append(float(node_attrs["z"]))

    z_range = max(z_coords) - min(z_coords)
    is_2d_data = z_range < 1e-10  # Use small tolerance for numerical precision

    if is_2d_data:
        print(
            f"Detected 2D data (z range: {z_range:.3f}). Masks and bboxes will be converted to 2D."
        )
    else:
        print(
            f"Detected 3D data (z range: {z_range:.3f}). Masks and bboxes will remain 3D."
        )

    # Process nodes first (all with parent_id = -1)
    print("Preparing node records...")
    node_records = []

    for rx_node_id in tqdm(rx_graph.node_indices(), desc="Processing nodes"):
        node_attrs = rx_graph[rx_node_id]

        # Get original node ID
        node_id = rx_to_original_id.get(rx_node_id, rx_node_id)

        # Extract node attributes from GEFF
        t = int(node_attrs["t"])
        z = float(node_attrs["z"])
        y = float(node_attrs["y"])
        x = float(node_attrs["x"])
        solution = bool(node_attrs["solution"])
        bbox = np.array(node_attrs["bbox"], dtype=np.int32)

        # Convert mask to proper numpy array for Node object
        mask_array = node_attrs["mask"]
        if not isinstance(mask_array, np.ndarray):
            mask_array = np.array(mask_array)
        mask_bool = np.ascontiguousarray(mask_array, dtype=bool)

        # Convert to 2D if data is detected as 2D
        if is_2d_data and mask_bool.ndim == 3:
            # For 2D data, take the first z-slice or sum across z if multiple slices
            if mask_bool.shape[0] == 1:
                mask_bool = mask_bool[0]  # Take the single z-slice
            else:
                mask_bool = np.any(mask_bool, axis=0)  # Combine all z-slices

        # Handle bbox similarly
        if is_2d_data and len(bbox) == 6:  # 3D bbox: [x1, y1, z1, x2, y2, z2]
            bbox = np.array(
                [bbox[1], bbox[2], bbox[4], bbox[5]]
            )  # Extract [x1, y1, x2, y2] -> [x_min, y_min, x_max, y_max]

        # Create Node object
        # Note: The vendored Node.__init__ sets mask and bbox to None when parent is None,
        # so we must set them after initialization
        node_obj = Node(h_node_index=-1, parent=None)
        node_obj.mask = mask_bool
        node_obj.bbox = bbox
        node_obj.area = int(np.sum(node_obj.mask))
        node_obj.centroid = (
            node_obj._centroid() if hasattr(node_obj, "_centroid") else None
        )

        # Convert to Ultrack node ID format (t0000xxx)
        node_id_ultrack_format = _generate_id(node_id, t, _max_segments_per_time)

        # Create database record
        node_record = {
            "id": node_id_ultrack_format,
            "t": t,
            "parent_id": -1,  # Initialize all as root nodes, will update from solution edges
            "hier_parent_id": -1,
            "t_node_id": 0,
            "t_hier_id": 0,
            "z": z,
            "y": y,
            "x": x,
            "z_shift": 0.0,
            "y_shift": 0.0,
            "x_shift": 0.0,
            "area": node_obj.area,
            "frontier": -1.0,
            "height": -1.0,
            "selected": solution,
            "pickle": node_obj,
            "features": None,
            "node_prob": -1.0,
        }
        node_records.append(node_record)

    edge_records = []
    # Create a mapping from node_id to index in node_records for quick updates
    node_id_to_index = {record["id"]: i for i, record in enumerate(node_records)}

    for edge_idx in tqdm(rx_graph.edge_indices(), desc="Processing edges"):
        # Get source and target rustworkx node IDs
        source_rx, target_rx = rx_graph.get_edge_endpoints_by_index(edge_idx)

        # Convert to original node IDs (1:N)
        source_id = rx_to_original_id.get(source_rx, source_rx)
        target_id = rx_to_original_id.get(target_rx, target_rx)

        # Convert to Ultrack node ID format (t0000xxx)
        source_id = _generate_id(
            source_id, int(rx_graph[source_rx]["t"]), _max_segments_per_time
        )
        target_id = _generate_id(
            target_id, int(rx_graph[target_rx]["t"]), _max_segments_per_time
        )

        # Get edge attributes
        edge_attrs = rx_graph.get_edge_data_by_index(edge_idx)

        # Check if this is a solution edge
        if edge_attrs.get("solution", False):
            # Source is parent, target is child - update the node record directly
            if target_id in node_id_to_index:
                target_index = node_id_to_index[target_id]
                node_records[target_index]["parent_id"] = source_id

        edge_record = {
            "source_id": source_id,
            "target_id": target_id,
        }
        edge_records.append(edge_record)

    # Insert nodes into database (now with correct parent IDs)
    with Session(engine) as session:
        session.bulk_insert_mappings(NodeDB, node_records)
        session.commit()
    print(f"✓ Inserted {len(node_records)} nodes")

    # Insert edges into database
    with Session(engine) as session:
        session.bulk_insert_mappings(LinkDB, edge_records)
        session.commit()
    print(f"✓ Inserted {len(edge_records)} edges")

    print(f"✓ Database saved to: {database_path}")
    print("Don't forget to create a metadata.toml file with the following content: ")
    print("shape = [ 5, 128, 128, 128] ([t, (z, ), y, x])")


@click.command()
@click.argument("geff_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output database path (default: <input_stem>_to_db.db)",
)
def convert_geff_to_db_cli(geff_path: Path, output: Path = None) -> None:
    """Convert GEFF file to ULTrack SQLite database (CLI version).

    Args:
        geff_path: Path to the input GEFF file
        output: Optional output database path
    """
    convert_geff_to_db(geff_path, output)


if __name__ == "__main__":
    convert_geff_to_db_cli()
