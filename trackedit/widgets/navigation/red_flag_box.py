import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QPushButton, QHBoxLayout, QLabel, QVBoxLayout
from trackedit.widgets.ClickableLabel import ClickableLabel
from .base_box import NavigationBox

class RedFlagBox(NavigationBox):

    update_chunk_from_frame_signal = Signal(int)

    def __init__(self, tracks_viewer, databasehandler):
        super().__init__("Red Flags", max_height=120)
        self.tracks_viewer = tracks_viewer
        self.databasehandler = databasehandler
        self.current_red_flag_index = 0

        # Create controls
        self.red_flag_counter = ClickableLabel("0/0")
        self.red_flag_counter.setFixedWidth(50)
        
        self.red_flag_prev_btn = QPushButton("<")
        self.red_flag_prev_btn.setFixedWidth(30)
        
        self.red_flag_next_btn = QPushButton(">")
        self.red_flag_next_btn.setFixedWidth(30)
        
        self.red_flag_ignore_btn = QPushButton("ignore")
        self.red_flag_info = QLabel("info")

        # Layout
        red_flag_layout = QVBoxLayout()
        red_flag_layout_row1 = QHBoxLayout()
        red_flag_layout_row1.addWidget(self.red_flag_prev_btn)
        red_flag_layout_row1.addWidget(self.red_flag_counter)
        red_flag_layout_row1.addWidget(self.red_flag_next_btn)
        red_flag_layout_row1.addWidget(self.red_flag_ignore_btn)
        
        red_flag_layout.addLayout(red_flag_layout_row1)
        red_flag_layout.addWidget(self.red_flag_info)
        self.layout.addLayout(red_flag_layout)

        # Connect buttons
        self.red_flag_prev_btn.clicked.connect(self.go_to_prev_red_flag)
        self.red_flag_next_btn.clicked.connect(self.go_to_next_red_flag)
        self.red_flag_ignore_btn.clicked.connect(self.ignore_red_flag)
        self.red_flag_counter.clicked.connect(self.goto_red_flag)
        self.tracks_viewer.tracks_updated.connect(self.update_red_flags)
        self.tracks_viewer.selected_nodes.list_updated.connect(self._check_selected_node_matches_red_flag)

    def update_red_flags(self):
        """Update the red flags and counter"""
        self.databasehandler.recompute_red_flags()
        self.update_red_flag_counter_and_info()

    def update_red_flag_counter_and_info(self):
        """Update the red flag label to show the current red flag index and total count."""
        total = len(self.databasehandler.red_flags)
        if total > 0:
            self.red_flag_counter.setText(f"{self.current_red_flag_index + 1}/{total}")
            df_rf = self.databasehandler.red_flags.iloc[[self.current_red_flag_index]]
            text = f"{df_rf.iloc[0].id} {df_rf.iloc[0].event} at t={df_rf.iloc[0].t}"
            self.red_flag_info.setText(text)
        else:
            self.red_flag_counter.setText("0/0")
            self.red_flag_info.setText("-")

    def go_to_next_red_flag(self):
        """Navigate to the next red flag in the list and jump to that timepoint."""
        total = len(self.databasehandler.red_flags)
        if total == 0:
            return
        self.current_red_flag_index = (self.current_red_flag_index + 1) % total
        self.goto_red_flag()

    def go_to_prev_red_flag(self):
        """Navigate to the previous red flag in the list and jump to that timepoint."""
        total = len(self.databasehandler.red_flags)
        if total == 0:
            return
        self.current_red_flag_index = (self.current_red_flag_index - 1) % total
        self.goto_red_flag()

    def goto_red_flag(self):
        """Jump to the time of the current red flag."""
        red_flag_time = int(self.databasehandler.red_flags.iloc[self.current_red_flag_index]["t"])
        self.update_chunk_from_frame_signal.emit(red_flag_time)

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
            self.red_flag_counter.setStyleSheet("color: gray;")
            self.red_flag_ignore_btn.setEnabled(False)
            return

        selected_node = selected_nodes[0]
        red_flag_ids = self.databasehandler.red_flags['id'].values
        
        try:
            index = np.where(red_flag_ids == selected_node)[0][0]
            # Found the node in red flags - update counter and remove grey
            self.current_red_flag_index = index
            self.red_flag_counter.setText(f"{index + 1}/{len(self.databasehandler.red_flags)}")
            self.red_flag_counter.setStyleSheet("")
            self.red_flag_ignore_btn.setEnabled(True)
        except IndexError:
            # Node not found in red flags - grey out counter
            self.red_flag_counter.setStyleSheet("color: gray;") 
            self.red_flag_ignore_btn.setEnabled(False)