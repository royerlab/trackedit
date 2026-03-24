"""Command line interface for trackedit."""

from pathlib import Path

import click

from trackedit.utils.crop import crop_database_in_time
from trackedit.utils.geff import convert_geff_to_db


@click.group()
@click.version_option()
def cli():
    """TrackeEdit: A tool for editing and converting tracking data."""


@cli.group()
def convert():
    """Convert between different tracking data formats."""


@convert.command("geff-to-db")
@click.argument("geff_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output database path (default: <input_stem>_from_geff.db)",
)
def geff_to_db(geff_path: Path, output: Path = None):
    """Convert GEFF file to Ultrack SQLite database.

    Args:
        geff_path: Path to the input GEFF file
        output: Optional output database path
    """
    convert_geff_to_db(geff_path, output)


@cli.command("crop-db")
@click.argument("source_db", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--max-t",
    required=True,
    type=int,
    help="Maximum time frame to include (inclusive).",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output database path (default: <input_stem>_t0-<max_t>.db)",
)
def crop_db(source_db: Path, max_t: int, output: Path = None):
    """Crop an Ultrack SQLite database to the first MAX_T frames."""
    if output is None:
        output = source_db.parent / f"{source_db.stem}_t0-{max_t}.db"
    crop_database_in_time(source_db, output, max_t)
    print(f"Cropped database written to {output}")


if __name__ == "__main__":
    cli()
