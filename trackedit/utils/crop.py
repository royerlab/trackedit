import shutil
import sqlite3
from pathlib import Path


def crop_database_in_time(source_db: Path, output_db: Path, max_t: int) -> None:
    """Create a time-cropped copy of an ultrack SQLite database.

    Copies nodes, links, and overlaps tables from the source database,
    keeping only entries up to and including time frame `max_t`.
    The gt_nodes and gt_links tables are copied as-is (assumed empty).

    Args:
        source_db: Path to the source .db file.
        output_db: Path where the cropped copy will be written.
        max_t: Maximum time frame to include (inclusive). Nodes with t <= max_t
               are kept; links and overlaps are filtered to only reference
               surviving node IDs.
    """
    shutil.copy2(source_db, output_db)

    conn = sqlite3.connect(output_db)
    try:
        # Remove nodes outside time range
        conn.execute("DELETE FROM nodes WHERE t > ?", (max_t,))

        # Collect surviving node IDs
        surviving_ids = {
            row[0] for row in conn.execute("SELECT id FROM nodes").fetchall()
        }

        # Remove links where either endpoint is gone
        all_links = conn.execute(
            "SELECT id, source_id, target_id FROM links"
        ).fetchall()
        link_ids_to_delete = [
            row[0]
            for row in all_links
            if row[1] not in surviving_ids or row[2] not in surviving_ids
        ]
        if link_ids_to_delete:
            conn.execute(
                f"DELETE FROM links WHERE id IN ({','.join('?' * len(link_ids_to_delete))})",
                link_ids_to_delete,
            )

        # Remove overlaps where either node_id or ancestor_id is gone
        all_overlaps = conn.execute(
            "SELECT rowid, node_id, ancestor_id FROM overlaps"
        ).fetchall()
        overlap_rowids_to_delete = [
            row[0]
            for row in all_overlaps
            if row[1] not in surviving_ids or row[2] not in surviving_ids
        ]
        if overlap_rowids_to_delete:
            conn.execute(
                f"DELETE FROM overlaps WHERE rowid IN ({','.join('?' * len(overlap_rowids_to_delete))})",
                overlap_rowids_to_delete,
            )

        conn.commit()
        conn.execute("VACUUM")
    finally:
        conn.close()


if __name__ == "__main__":
    from trackedit.cli import cli

    cli()
