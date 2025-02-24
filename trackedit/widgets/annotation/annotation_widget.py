import napari
import numpy as np
from motile_tracker.data_views import TracksViewer   
from trackedit.DatabaseHandler import DatabaseHandler
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
)
from trackedit.widgets.annotation.todo_box import TodoAnnotationBox

class AnnotationWidget(QWidget):

    change_chunk = Signal(str)
    goto_frame = Signal(int)

    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        super().__init__()

        self.viewer = viewer
        self.databasehandler = databasehandler
        self.tracks_viewer = TracksViewer.get_instance(self.viewer)
        self.current_todoannotation_index = 0

        # Create boxes
        self.todo_box = TodoAnnotationBox(self.tracks_viewer, self.databasehandler)

        # Forward signals
        # self.time_box.change_chunk.connect(self.change_chunk)
        # self.time_box.goto_frame.connect(self.goto_frame)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.todo_box)
        # main_layout.addWidget(self.red_flag_box)
        # main_layout.addWidget(self.division_box)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 2, 10, 2)
        self.setLayout(main_layout)