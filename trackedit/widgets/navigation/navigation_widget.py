import napari
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QVBoxLayout, QWidget

from motile_tracker.data_views import TracksViewer
from trackedit.DatabaseHandler import DatabaseHandler
from trackedit.widgets.navigation.division_box import DivisionBox
from trackedit.widgets.navigation.red_flag_box import RedFlagBox
from trackedit.widgets.navigation.time_box import TimeBox


class NavigationWidget(QWidget):

    change_chunk = Signal(str)
    goto_frame = Signal(int)

    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        super().__init__()
        self.setObjectName("NavigationWidget")
        self.viewer = viewer
        self.databasehandler = databasehandler
        self.tracks_viewer = TracksViewer.get_instance(self.viewer)
        self.current_red_flag_index = 0
        self.current_division_index = 0

        # Create boxes
        self.time_box = TimeBox(self.viewer, self.databasehandler)
        self.red_flag_box = RedFlagBox(self.tracks_viewer, self.databasehandler)
        self.division_box = DivisionBox(self.tracks_viewer, self.databasehandler)

        # Forward signals
        self.time_box.change_chunk.connect(self.change_chunk)
        self.time_box.goto_frame.connect(self.goto_frame)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.time_box)
        main_layout.addWidget(self.red_flag_box)
        main_layout.addWidget(self.division_box)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 2, 10, 2)
        self.setLayout(main_layout)
