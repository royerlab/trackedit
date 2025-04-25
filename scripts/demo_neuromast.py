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
working_directory = Path(
    "/home/teun.huijben/Documents/data/Akila/20241003/neuromast4_t851/adjusted/"
)
# name of the database file to start from, or "latest" to start from the latest version, defaults to "data.db"
db_filename_start = "latest"
# maximum number of frames display, defaults to None (use all frames)
tmax = 600
# (Z),Y,X, defaults to (1, 1, 1)
scale = (2.31, 1, 1)
# overwrite existing database/changelog, defaults to False (not used when db_filename_start is "latest")
allow_overwrite = False

# OPTIONAL: imaging data
# imaging_zarr_file = (
#     "/hpc/projects/jacobo_group/iSim_processed_files/steady_state_timelapses/"
#     "20241003_2dpf_myo6b_bactin_GFP_she_h2b_gfp_cldnb_lyn_mScarlet/"
#     "46hpf_fish1_1/2_deconvolution_and_registration/"
#     "deconvolved_and_registered_corrected_cropped850.zarr"
# )
# imaging_channel = "0/4/0/0"
imaging_zarr_file = None
imaging_channel = None

# OPTIONAL: annotation mapping (default is neuromast cell types)
# annotation_mapping = {
#     1: {"name": "hair", "color": [0.0, 1.0, 0.0, 1.0]},  # green
#     2: {"name": "support", "color": [1.0, 0.1, 0.6, 1.0]},  # pink
#     3: {"name": "mantle", "color": [0.0, 0.0, 0.9, 1.0]},  # blue
# }
annotation_mapping = None
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
        annotation_mapping=annotation_mapping,
    )
