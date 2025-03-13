import sys
import warnings
from pathlib import Path

import napari
import numpy as np

# Databases saved with numpy>2 need np._core.numeric, which is not available in numpy<2, hence the following hack
sys.modules["numpy._core.numeric"] = np.core.numeric

from motile_tracker.data_model.actions import (
    AddEdges,
    AddNodes,
    DeleteEdges,
    DeleteNodes,
)
from motile_tracker.data_views import TracksViewer
from trackedit.DatabaseHandler import DatabaseHandler
from trackedit.motile_overwrites import (
    create_db_add_edges,
    create_db_add_nodes,
    create_db_delete_edges,
    create_db_delete_nodes,
    create_tracks_viewer_and_segments_refresh,
)
from trackedit.TrackEditClass import TrackEditClass
from trackedit.utils.utils import wrap_default_widgets_in_tabs

warnings.filterwarnings("ignore", category=FutureWarning, message=".*qt_viewer.*")

# **********INPUTS*********
working_directory = Path(
    "/home/teun.huijben/Documents/data/Akila/20241003/neuromast4_t851/adjusted/"
)
# working_directory = Path("/Users/teun.huijben/Documents/data/Akila/20241003_neuromast4/adjusted/")
# working_directory = Path("/hpc/projects/jacobo_group/iSim_processed_files/steady_state_timelapses/20241003_2dpf_myo6b_bactin_GFP_she_h2b_gfp_cldnb_lyn_mScarlet/46hpf_fish1_1/4_tracking/database/")
# path to the working directory that contains the database file AND metadata.toml
db_filename_old = "data.db"  # name of the database file to start from
Tmax = 600  # maximum number of frames display (crop the data to this number of frames)
scale = (2.31, 1, 1)  # (Z),Y,X
layer_name = "ultrack"  # name of the layer in napari
allow_overwrite = True  # overwrite existing database/changelog

# OPTIONAL: imaging data
# imaging_zarr_file = "/hpc/projects/jacobo_group/iSim_processed_files/steady_state_timelapses/20241003_2dpf_myo6b_bactin_GFP_she_h2b_gfp_cldnb_lyn_mScarlet/46hpf_fish1_1/2_deconvolution_and_registration/deconvolved_and_registered_corrected_cropped850.zarr"
# imaging_channel = "0/4/0/0"
imaging_zarr_file = None
imaging_channel = None
# *************************


def main():
    DB_handler = DatabaseHandler(
        db_filename_old=db_filename_old,
        working_directory=working_directory,
        Tmax=Tmax,
        scale=scale,
        name="ultrack",
        allow_overwrite=allow_overwrite,
        imaging_zarr_file=imaging_zarr_file,
        imaging_channel=imaging_channel,
    )

    # overwrite some motile functions
    DeleteNodes._apply = create_db_delete_nodes(DB_handler)
    DeleteEdges._apply = create_db_delete_edges(DB_handler)
    AddEdges._apply = create_db_add_edges(DB_handler)
    AddNodes._apply = create_db_add_nodes(DB_handler)
    TracksViewer._refresh = create_tracks_viewer_and_segments_refresh(
        layer_name=layer_name
    )

    # open napari with TrackEdit
    viewer = napari.Viewer()
    _trackeditclass = TrackEditClass(viewer, databasehandler=DB_handler)
    viewer.dims.ndisplay = 3  # 3D view
    wrap_default_widgets_in_tabs(viewer)
    viewer.dims.current_step = (2, *viewer.dims.current_step[1:])
    napari.run()


if __name__ == "__main__":
    main()
