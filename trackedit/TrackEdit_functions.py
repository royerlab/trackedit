import napari
import pandas as pd

from motile_tracker.application_menus.editing_menu import EditingMenu
from motile_tracker.data_views import TracksViewer, TreeWidget   
from motile_tracker.data_model.solution_tracks import SolutionTracks

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QWidget,
)

def remove_past_parents_from_df(df2):

    df2.loc[:,"t"] = df2["t"] - df2["t"].min()

    #find the first time point
    min_time = 0

    # Set all parent_id values to -1 for the first time point
    df2.loc[df2["t"] == min_time, "parent_id"] = -1

    #find the tracks with parents at the first time point
    tracks_with_parents = df2.loc[(df2["t"] == min_time) & (df2["parent_track_id"] != -1), "track_id"]
    track_ids_to_update = set(tracks_with_parents)

    # update the parent_track_id to -1 for the tracks with parents at the first time point
    df2.loc[df2["track_id"].isin(track_ids_to_update), "parent_track_id"] = -1
    return df2

class trackEdit_sidebar(QWidget):
    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        layout = QHBoxLayout()

        #Define the buttons
        self.time_prev_btn = QPushButton("prev T")
        self.time_prev_btn.clicked.connect(self.press_prev)
        self.time_next_btn = QPushButton("next T")
        self.time_next_btn.clicked.connect(self.press_next)
        
        layout.addWidget(self.time_prev_btn)
        layout.addWidget(self.time_next_btn)
        self.setLayout(layout)
        self.setMaximumHeight(300)

    def press_next(self):
        print('next')

    def press_prev(self):
        print('prev')

def add_motile_widgets(viewer: napari.Viewer):
    """Add the Tree Widget and the Results List dock widgets. If you want to edit,
    You can also add the EditWidget and/or the MainApp widget, which is all the 
    widgets combined.

    Args:
        viewer (napari.Viewer): The napari viweer
    """
    # tracksviewer = TracksViewer.get_instance(viewer)
    viewer.window.add_dock_widget(TreeWidget(viewer), area="bottom",name="TreeWidget")
    # # # viewer.window.add_dock_widget(tracksviewer.tracks_list)
    viewer.window.add_dock_widget(EditingMenu(viewer),area="right",name="EditingMenu")
    viewer.window.add_dock_widget(trackEdit_sidebar(viewer), area='right', name='TrackEdit')

def add_tracks(viewer: napari.Viewer, nxgraph, segmentation, name: str):
    """Add a solution set of tracks to the tracks viewer results list

    Args:
        viewer (napari.Viewer): the napari viewer
        tracker (ultrack.Tracker): the ultrack tracker containing the solution
        name (str): the display name of the solution tracks
    """

    # create tracks object
    tracks = SolutionTracks(
        nxgraph,
        segmentation = segmentation,
        pos_attr=("z","y", "x"),
        time_attr="t",
        scale = [1,4,1,1],
    )
    
    tracksviewer = TracksViewer.get_instance(viewer)
    tracksviewer.tracks_list.add_tracks(tracks, name)