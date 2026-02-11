import sys
import warnings
from pathlib import Path

import numpy as np

from trackedit.run import run_trackedit

# Databases saved with numpy>2 need np._core.numeric, which is not available in numpy<2, hence the following hack
sys.modules["numpy._core.numeric"] = np.core.numeric

warnings.filterwarnings("ignore", category=FutureWarning, message=".*qt_viewer.*")

# **********INPUTS*********
# path to the working directory that contains the database file AND metadata.toml:
working_directory = Path("/home/teun.huijben/Documents/data/Thibaut/masks_on_geff/")
# name of the database file to start from, or "latest" to start from the latest version, defaults to "data.db"
db_filename_start = "latest"
# maximum number of frames display, defaults to None (use all frames)
tmax = 100
# (Z),Y,X, defaults to (1, 1, 1)
scale = (1.625, 0.40625, 0.40625)
# overwrite existing database/changelog, defaults to False (not used when db_filename_start is "latest")
allow_overwrite = False

# OPTIONAL: imaging data
imaging_zarr_file = (
    "/hpc/projects/group.royer/people/teun.huijben/data/Thibault/4th_exp/first_fov.zarr"
)
imaging_channel = "0"
imaging_layer_names = ["dense", "sparse"]

# OPTIONAL: annotation mapping (default is neuromast cell types)
# annotation_mapping = {
#     1: {"name": "hair", "color": [0.0, 1.0, 0.0, 1.0]},  # green
#     2: {"name": "support", "color": [1.0, 0.1, 0.6, 1.0]},  # pink
#     3: {"name": "mantle", "color": [0.0, 0.0, 0.9, 1.0]},  # blue
# }
annotation_mapping = None

# OPTIONAL: InstanSeg model for interactive cell segmentation
# Enable this to add cells via InstanSeg inference instead of spherical masks
flag_allow_adding_instanseg_cell = True
instanseg_model_path = (
    "/hpc/projects/group.royer/people/teun.huijben/data/Thibault/model_96.pt"
)
instanseg_device = None  # 'cuda', 'cpu', or None for auto-detect
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
        imaging_layer_names=imaging_layer_names,
        annotation_mapping=annotation_mapping,
        flag_allow_adding_spherical_cell=True,
        adding_spherical_cell_radius=10,
        flag_remove_red_flags_at_edge=True,
        remove_red_flags_at_edge_threshold=10,
        flag_allow_adding_instanseg_cell=flag_allow_adding_instanseg_cell,
        instanseg_model_path=instanseg_model_path,
        instanseg_device=instanseg_device,
    )
