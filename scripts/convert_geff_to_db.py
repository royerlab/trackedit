"""
Script to convert GEFF files to Ultrack SQLite databases for the cellmotv1 tracking challenge.

For each experiment folder:
  - Runs: trackedit convert geff-to-db <experiment>.geff -o data.db
  - Writes: metadata.toml with shape = [ 100, 64, 256, 256,]

Usage:
  python convert_geff_to_db.py              # test mode: processes one experiment, output to Desktop
  python convert_geff_to_db.py --all        # processes all experiments, output next to geff/zarr
"""

import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(
    "/hpc/projects/group.royer/people/thibaut.goldsborough/tracking-challenge/cellmotv1"
)

# Used only in test mode (no write access to BASE_DIR)
TEST_OUTPUT_DIR = Path("/home/teun.huijben/Desktop/tracking-challenge/cellmotv1")

SHAPE = [100, 64, 256, 256]

METADATA_CONTENT = f"shape = {SHAPE}\n"


def process_experiment(experiment_dir: Path, out_dir: Path) -> bool:
    geff_files = list(experiment_dir.glob("*.geff"))
    if not geff_files:
        print(f"  [SKIP] No .geff file found in {experiment_dir}")
        return False

    geff_path = geff_files[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    db_output = out_dir / "data.db"
    metadata_path = out_dir / "metadata.toml"

    print(f"  Converting: {geff_path.name}")
    result = subprocess.run(
        ["trackedit", "convert", "geff-to-db", str(geff_path), "-o", str(db_output)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"  [ERROR] Conversion failed:\n{result.stderr}")
        return False

    print(f"  Written:    {db_output}")

    metadata_path.write_text(METADATA_CONTENT)
    print(f"  Written:    {metadata_path}")

    return True


def main():
    run_all = "--all" in sys.argv

    date_folders = sorted(BASE_DIR.iterdir())

    if run_all:
        experiments = [
            exp
            for date_folder in date_folders
            for exp in sorted(date_folder.iterdir())
            if exp.is_dir()
        ]
        print(f"Processing all {len(experiments)} experiments...\n")
        pairs = [(exp, exp) for exp in experiments]
    else:
        # Test mode: first experiment of the first folder, write to Desktop
        first_folder = date_folders[0]
        exp = sorted(first_folder.iterdir())[0]
        experiments = [exp]
        relative = exp.relative_to(BASE_DIR)
        out = TEST_OUTPUT_DIR / relative
        pairs = [(exp, out)]
        print("TEST MODE: processing 1 experiment")
        print(f"  Input:  {exp}")
        print(f"  Output: {out}\n")

    success = 0
    for exp, out in pairs:
        print(f"[{exp.parent.name}/{exp.name}]")
        if process_experiment(exp, out):
            success += 1

    print(f"\nDone: {success}/{len(experiments)} experiments converted successfully.")


if __name__ == "__main__":
    main()
