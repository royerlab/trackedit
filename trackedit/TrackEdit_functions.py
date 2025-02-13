import napari
import pandas as pd

from motile_tracker.application_menus.editing_menu import EditingMenu
from motile_tracker.data_views import TracksViewer, TreeWidget   
from motile_tracker.data_model.solution_tracks import SolutionTracks

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
)
from PyQt5.QtCore import Qt
from qtpy.QtCore import Signal
from trackedit.DatabaseHandler import DatabaseHandler

class TrackEditSidebar(QWidget):

    change_chunk = Signal(str)

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.tracks_viewer = TracksViewer.get_instance(viewer)

        #Define the buttons
        self.time_prev_btn = QPushButton("prev (<)")
        self.time_prev_btn.clicked.connect(self.press_prev)
        self.time_next_btn = QPushButton("next (>)")
        self.time_next_btn.clicked.connect(self.press_next)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.time_prev_btn)
        button_layout.addWidget(self.time_next_btn)

        #Define the time window label
        label = "temp. label"
        self.chunk_label = QLabel(label)

        label_layout = QVBoxLayout()
        label_layout.addWidget(self.chunk_label, alignment=Qt.AlignCenter)

        #Define entire widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(self._title_widget())
        main_layout.addLayout(label_layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)
        self.setMaximumHeight(100)

    def press_prev(self):
        self.change_chunk.emit('prev')
        # self.update_label()

    def press_next(self):
        self.change_chunk.emit('next')
        # self.update_label()

    def update_label(self):
        time_window = self.tracks_viewer.tracks.segmentation.time_window
        label = f"time window ({time_window[0]} : {time_window[1]-1})"
        self.chunk_label.setText(label)


    def _title_widget(self) -> QWidget:
        """Create the title and intro paragraph widget, with links to docs

        Returns:
            QWidget: A widget introducing the motile tracker and linking to docs
        """
        richtext = r"""<h3>Tracking with blablabla</h3>""" 
        label = QLabel(richtext)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        return label
    

class TrackEditClass():
    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        self.viewer = viewer

        self.TreeWidget = TreeWidget(self.viewer)
        self.TrackEditSidebar = TrackEditSidebar(self.viewer)
        self.EditingMenu = EditingMenu(self.viewer)

        self.viewer.window.add_dock_widget(self.TreeWidget, area="bottom",name="TreeWidget")
        self.viewer.window.add_dock_widget(self.TrackEditSidebar, area='right', name='TrackEditSidebar')
        self.viewer.window.add_dock_widget(self.EditingMenu,area="right",name="EditingMenu")

        #Todo: provide entire DB_handler
        self.databasehandler = databasehandler
        self.TrackEditSidebar.change_chunk.connect(self.update_chunk)

        self.add_tracks()

    def add_tracks(self):
        """Add a solution set of tracks to the tracks viewer results list

        Args:
            tracker (ultrack.Tracker): the ultrack tracker containing the solution
            name (str): the display name of the solution tracks
        """

        # create tracks object
        tracks = SolutionTracks(
            graph = self.databasehandler.nxgraph,
            segmentation = self.databasehandler.segments,
            pos_attr=("z","y", "x"),
            time_attr="t",
            scale = [1,4,1,1],
        )

        # add tracks to viewer
        tracksviewer = TracksViewer.get_instance(self.viewer)
        tracksviewer.tracks_list.add_tracks(tracks,name=self.databasehandler.name)
        self.viewer.layers.selection.active = self.viewer.layers[self.databasehandler.name+'_seg']   #select segmentation layer

        #update label in TrackEditSidebar
        self.TrackEditSidebar.update_label()

        self.check_button_validity()

        #ToDo: check if all tracks are added or overwritten
    

    def update_chunk(self, direction: str):
        cur_chunk = self.databasehandler.time_chunk

        #change the time chunk index
        if direction == 'prev':
            new_chunk = cur_chunk - 1
        elif direction == 'next':
            new_chunk = cur_chunk + 1

        #check if the new chunk is within the limits
        if new_chunk < 0:
            new_chunk = 0
        elif new_chunk == self.databasehandler.num_time_chunks:
            new_chunk = self.databasehandler.num_time_chunks - 1

        self.databasehandler.set_time_chunk(new_chunk)
        self.add_tracks()

    def check_button_validity(self):
        #enable/disable buttons if on first/last chunk
        chunk = self.databasehandler.time_chunk
        if chunk == 0:
            self.TrackEditSidebar.time_prev_btn.setEnabled(False)
        else:
            self.TrackEditSidebar.time_prev_btn.setEnabled(True)

        if chunk == self.databasehandler.num_time_chunks - 1:
            self.TrackEditSidebar.time_next_btn.setEnabled(False)
        else:
            self.TrackEditSidebar.time_next_btn.setEnabled(True)

