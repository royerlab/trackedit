import napari
import numpy as np
from motile_tracker.data_views import TracksViewer   
from trackedit.DatabaseHandler import DatabaseHandler
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
)
from .navigation.time_box import TimeBox
from .navigation.red_flag_box import RedFlagBox
from .navigation.division_box import DivisionBox

class NavigationWidget(QWidget):

    change_chunk = Signal(str)
    goto_frame = Signal(int)

    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        super().__init__()

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

    # Delegate methods to appropriate boxes
    def set_time_slider(self, chunk_frame):
        self.time_box.set_time_slider(chunk_frame)

    def update_time_label(self):
        self.time_box.update_time_label()

    def check_navigation_button_validity(self):
        self.time_box.check_navigation_button_validity()

    def update_chunk_label(self):
        self.time_box.update_chunk_label()

    #===============================================
    # Red flags
    #===============================================

    def update_red_flags(self):
        self.databasehandler.recompute_red_flags()
        self.update_red_flag_counter_and_info()

    def update_red_flag_counter_and_info(self):
        """Update the red flag label to show the current red flag index and total count."""
        print(f"updating red flag counter and info for {self.current_red_flag_index}")
        total = len(self.databasehandler.red_flags)
        if total > 0:
            self.red_flag_box.red_flag_counter.setText(f"{self.current_red_flag_index + 1}/{total}")
            df_rf = self.databasehandler.red_flags.iloc[[self.current_red_flag_index]]
            text = f"{df_rf.iloc[0].id} {df_rf.iloc[0].event} at t={df_rf.iloc[0].t}"
            self.red_flag_box.red_flag_info.setText(text)
        else:
            self.red_flag_box.red_flag_counter.setText("0/0")
            self.red_flag_box.red_flag_info.setText("-")

    def go_to_next_red_flag(self):
        """Navigate to the next red flag in the list and jump to that timepoint."""
        total = len(self.databasehandler.red_flags)
        if total == 0:
            return
        self.current_red_flag_index = (self.current_red_flag_index + 1) % total
        print(f"going to next red flag: {self.current_red_flag_index} of total: {total}")
        self.goto_red_flag()

    def go_to_prev_red_flag(self):
        """Navigate to the previous red flag in the list and jump to that timepoint."""
        total = len(self.databasehandler.red_flags)
        if total == 0:
            return
        self.current_red_flag_index = (self.current_red_flag_index - 1) % total
        print(f"going to prev red flag: {self.current_red_flag_index} of total: {total}")   
        self.goto_red_flag()

    def goto_red_flag(self):
        """Jump to the time of the current red flag."""
        red_flag_time = int(self.databasehandler.red_flags.iloc[self.current_red_flag_index]["t"])
        self.goto_frame.emit(red_flag_time)  # Changed to emit signal instead of direct call

        # Update the selected nodes in the TreeWidget
        label = self.databasehandler.red_flags.iloc[self.current_red_flag_index]["id"]
        self.tracks_viewer.selected_nodes._list = []
        self.tracks_viewer.selected_nodes._list.append(label)
        self.tracks_viewer.selected_nodes.list_updated.emit()

        self.update_red_flag_counter_and_info()

    def ignore_red_flag(self):
        """Ignore the current red flag and remove it from the list."""
        id = self.databasehandler.red_flags.iloc[self.current_red_flag_index]["id"]
        self.databasehandler.seg_ignore_red_flag(id)

        if self.current_red_flag_index >= len(self.databasehandler.red_flags):
            self.current_red_flag_index = self.current_red_flag_index - 1
        self.goto_red_flag()

    def _check_selected_node_matches_red_flag(self):
        """Check if the selected node matches the current red flag label."""
        selected_nodes = self.tracks_viewer.selected_nodes._list

        # If no nodes selected or multiple nodes selected, grey out counter
        if len(selected_nodes) != 1:
            self.red_flag_box.red_flag_counter.setStyleSheet("color: gray;")
            return

        selected_node = selected_nodes[0]
        red_flag_ids = self.databasehandler.red_flags['id'].values
        
        try:
            index = np.where(red_flag_ids == selected_node)[0][0]
            # Found the node in red flags - update counter and remove grey
            self.current_red_flag_index = index
            self.red_flag_box.red_flag_counter.setText(f"{index + 1}/{len(self.databasehandler.red_flags)}")
            self.red_flag_box.red_flag_counter.setStyleSheet("")
        except IndexError:
            # Node not found in red flags - grey out counter
            self.red_flag_box.red_flag_counter.setStyleSheet("color: gray;")

    #===============================================
    # Divisions
    #===============================================

    def update_divisions(self):
        """Update the divisions and the division counter"""
        self.databasehandler.recompute_divisions()
        self.update_division_counter()

    def update_division_counter(self):
        """Update the division counter to show the current division index and total count."""
        total = len(self.databasehandler.divisions)
        if total > 0:
            self.division_box.division_counter.setText(f"{self.current_division_index + 1}/{total}")
        else:
            self.division_box.division_counter.setText("0/0")

    def go_to_next_division(self):
        """Navigate to the next division in the list and jump to that timepoint."""
        total = len(self.databasehandler.divisions)
        if total == 0:
            return
        self.current_division_index = (self.current_division_index + 1) % total
        self.goto_division()

    def go_to_prev_division(self):
        """Navigate to the previous division in the list and jump to that timepoint."""
        total = len(self.databasehandler.divisions)
        if total == 0:
            return
        self.current_division_index = (self.current_division_index - 1) % total
        self.goto_division()

    def goto_division(self):
        """Jump to the time of the current division."""
        division_time = int(self.databasehandler.divisions.iloc[self.current_division_index]["t"])
        self.goto_frame.emit(division_time)  # Use existing signal

        #update the selected nodes in the TreeWidget
        label = self.databasehandler.divisions.iloc[self.current_division_index]["id"]
        self.tracks_viewer.selected_nodes._list = []
        self.tracks_viewer.selected_nodes._list.append(label)
        self.tracks_viewer.selected_nodes.list_updated.emit()

        self.update_division_counter()

    def _check_selected_node_matches_division(self):
        """Check if the selected node matches the current division label."""
        selected_nodes = self.tracks_viewer.selected_nodes._list

        if len(selected_nodes) != 1:
            self.division_box.division_counter.setStyleSheet("color: gray;")
            return

        selected_node = selected_nodes[0]
        division_ids = self.databasehandler.divisions['id'].values
        
        try:
            index = np.where(division_ids == selected_node)[0][0]
            self.current_division_index = index
            self.division_box.division_counter.setText(f"{index + 1}/{len(self.databasehandler.divisions)}")
            self.division_box.division_counter.setStyleSheet("")
        except IndexError:
            self.division_box.division_counter.setStyleSheet("color: gray;")