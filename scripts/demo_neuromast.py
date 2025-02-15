
import napari
from pathlib import Path
from trackedit.motile_overwrites import *
from trackedit.TrackEdit_functions import *
from trackedit.DatabaseHandler import DatabaseHandler
from motile_tracker.data_model.actions import AddEdges, DeleteNodes, DeleteEdges, AddNodes

#**********INPUTS*********
working_directory = Path('/home/teun.huijben/Documents/data/Akila/20241003/neuromast4_t851/adjusted/')
# working_directory = Path('/Users/teun.huijben/Documents/data/Akila/20241003_neuromast4/adjusted/')
db_filename_old = 'data.db'
data_shape_full = [600,73,1024,1024]      #T,(Z),Y,X       (851,73,1024,1024)
scale = (4,1,1)
layer_name = 'ultrack'
allow_overwrite = True      #overwrite existing database/changelog
#*************************

def main():
    DB_handler = DatabaseHandler(
                    db_filename_old = db_filename_old,
                    working_directory = working_directory,
                    data_shape_full = data_shape_full,
                    z_scale = scale[0],
                    name = 'ultrack',
                    allow_overwrite = allow_overwrite)

    #overwrite some motile functions
    DeleteNodes._apply = create_db_delete_nodes(DB_handler)
    DeleteEdges._apply = create_db_delete_edges(DB_handler)
    AddEdges._apply = create_db_add_edges(DB_handler)
    AddNodes._apply = create_db_add_nodes(DB_handler)
    TracksViewer._refresh = create_tracks_viewer_and_segments_refresh(layer_name=layer_name)

    #open napari with TrackEdit
    viewer = napari.Viewer()
    trackeditclass = TrackEditClass(viewer, databasehandler = DB_handler)
    viewer.dims.ndisplay = 3    #3D view
    napari.run()

if __name__ == "__main__":
    main()