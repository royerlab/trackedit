import napari
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from motile_tracker.data_views import TracksViewer
from trackedit.DatabaseHandler import DatabaseHandler
from trackedit.widgets.annotation.toannotate_box import ToAnnotateBox


class AnnotationWidget(QWidget):

    change_chunk = Signal(str)
    goto_frame = Signal(int)

    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        super().__init__()

        self.viewer = viewer
        self.databasehandler = databasehandler
        self.tracks_viewer = TracksViewer.get_instance(self.viewer)

        # Create boxes
        self.toannotate_box = ToAnnotateBox(self.tracks_viewer, self.databasehandler)

        # Forward signals
        self.toannotate_box.refresh_annotation_layer.connect(
            self.refresh_annotation_layer
        )

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.toannotate_box)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 2, 10, 2)
        self.setLayout(main_layout)

    def refresh_annotation_layer(self):
        self.viewer.layers["annotations"].data.force_refill()
        self.viewer.layers["annotations"].refresh()
