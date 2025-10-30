"""Command line interface for trackedit."""

import click
from pathlib import Path

from trackedit.utils.geff import convert_geff_to_db


@click.group()
@click.version_option()
def cli():
    """TrackeEdit: A tool for editing and converting tracking data."""
    pass


@cli.group()
def convert():
    """Convert between different tracking data formats."""
    pass


@convert.command("geff-to-db")
@click.argument("geff_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", 
    type=click.Path(path_type=Path),
    help="Output database path (default: <input_stem>_from_geff.db)"
)
def geff_to_db(geff_path: Path, output: Path = None):
    """Convert GEFF file to ULTrack SQLite database.
    
    Args:
        geff_path: Path to the input GEFF file
        output: Optional output database path
    """
    convert_geff_to_db(geff_path, output)


if __name__ == "__main__":
    cli()