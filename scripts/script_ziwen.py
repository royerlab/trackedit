import operator
import sys
import warnings
from pathlib import Path

import numpy as np
from ultrack.core.database import NodeDB

from trackedit.run import run_trackedit

# Databases saved with numpy>2 need np._core.numeric, which is not available in numpy<2, hence the following hack
sys.modules["numpy._core.numeric"] = np.core.numeric

warnings.filterwarnings("ignore", category=FutureWarning, message=".*qt_viewer.*")

# **********INPUTS*********
# path to the working directory that contains the database file AND metadata.toml:
working_directory = Path("/home/teun.huijben/Documents/data/Ziwen/2025_07_24_A549/")
# name of the database file to start from, or "latest" to start from the latest version, defaults to "data.db"
db_filename_start = "latest"
# maximum number of frames display, defaults to None (use all frames)
tmax = 66
# (Z),Y,X, defaults to (1, 1, 1)
scale = (0.1494, 0.1494)
# overwrite existing database/changelog, defaults to False (not used when db_filename_start is "latest")
allow_overwrite = False

# OPTIONAL: imaging data
imaging_zarr_file = (
    "/hpc/projects/intracellular_dashboard/organelle_dynamics/rerun/"
    "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV/2-assemble/"
    "2025_07_24_A549_SEC61_TOMM20_G3BP1_ZIKV.zarr/A/1/000000/"
)
imaging_channel = "0"
image_z_slice = (
    20  # (only look at specific z-slice of 3D stack, because tracking data was 2D)
)
image_translate = (
    0,
    442 * 0.15,  # (taken from 2-assemble/concatenate_cropped.yml)
    161 * 0.15,  # (taken from 2-assemble/concatenate_cropped.yml)
)  # (t,y,x) because we use image_z_slice=20, so the z-axis is collapsed

# imaging_zarr_file = None
# imaging_channel = None
# image_translate = None

# OPTIONAL: annotation mapping (default is neuromast cell types)
annotation_mapping = {
    1: {"name": "normal", "color": [0.0, 1.0, 0.0, 1.0]},  # green
    2: {"name": "infected", "color": [1.0, 0.0, 0.0, 1.0]},  # red
    3: {"name": "other", "color": [0.5, 0.5, 0.5, 1.0]},  # grey
}
# annotation_mapping = None


# filter the database segments on pixel coordinates
# (if only displaying a crop of the full dataset)
coordinate_filters = [
    (NodeDB.x, operator.lt, 1280),  # lt = less than (<), coorddinates in db/pixel units
    (NodeDB.x, operator.gt, 161),  # gt = greater than (>)
    (NodeDB.y, operator.lt, 1186),
    (NodeDB.y, operator.gt, 442),
]
# *************************

if __name__ == "__main__":
    run_trackedit(
        working_directory=working_directory,
        db_filename=db_filename_start,
        tmax=tmax,
        scale=scale,
        allow_overwrite=allow_overwrite,
        imaging_zarr_file=imaging_zarr_file,
        imaging_channel=imaging_channel,
        image_z_slice=image_z_slice,
        image_translate=image_translate,
        annotation_mapping=annotation_mapping,
        coordinate_filters=coordinate_filters,
    )
