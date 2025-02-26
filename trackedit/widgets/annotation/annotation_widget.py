import napari
import numpy as np
from motile_tracker.data_views import TracksViewer   
from trackedit.DatabaseHandler import DatabaseHandler
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
)
from trackedit.widgets.annotation.todo_box import ToAnnotateBox

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
        self.todo_box = ToAnnotateBox(self.tracks_viewer, self.databasehandler)

        # Forward signals
        self.todo_box.refresh_annotation_layer.connect(self.refresh_annotation_layer)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.todo_box)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 2, 10, 2)
        self.setLayout(main_layout)

    def refresh_annotation_layer(self):
        self.viewer.layers['annotations'].data.force_refill()
        self.viewer.layers['annotations'].refresh()