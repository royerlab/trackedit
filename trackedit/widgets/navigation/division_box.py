import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QPushButton

from trackedit.widgets.base_box import NavigationBox
from trackedit.widgets.ClickableLabel import ClickableLabel


class DivisionBox(NavigationBox):

    update_chunk_from_frame_signal = Signal(int)

    def __init__(self, tracks_viewer, databasehandler):
        super().__init__("Divisions", max_height=80)
        self.setObjectName("DivisionBox")
        self.tracks_viewer = tracks_viewer
        self.databasehandler = databasehandler
        self.current_division_index = 0

        # Create controls
        self.division_counter = ClickableLabel("0/0")
        self.division_counter.setFixedWidth(50)

        self.division_prev_btn = QPushButton("<")
        self.division_prev_btn.setFixedWidth(30)

        self.division_next_btn = QPushButton(">")
        self.division_next_btn.setFixedWidth(30)

        # Layout
        division_layout = QHBoxLayout()
        division_layout.addWidget(self.division_prev_btn)
        division_layout.addWidget(self.division_counter)
        division_layout.addWidget(self.division_next_btn)

        self.layout.addLayout(division_layout)
        self.layout.setAlignment(division_layout, Qt.AlignLeft)

        # Connect buttons
        self.division_prev_btn.clicked.connect(self.go_to_prev_division)
        self.division_next_btn.clicked.connect(self.go_to_next_division)
        self.division_counter.clicked.connect(self.goto_division)
        self.tracks_viewer.tracks_updated.connect(self.update_divisions)
        self.tracks_viewer.selected_nodes.list_updated.connect(
            self._check_selected_node_matches_division
        )

    def update_divisions(self):
        """Update the divisions and the division counter"""
        self.databasehandler.recompute_divisions()
        self.update_division_counter()

    def update_division_counter(self):
        """Update the division counter to show the current division index and total count."""
        total = len(self.databasehandler.divisions)
        if total > 0:
            self.division_counter.setText(f"{self.current_division_index + 1}/{total}")
        else:
            self.division_counter.setText("0/0")

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
        # Check if there are any divisions before trying to access them
        if self.databasehandler.divisions.empty:
            return

        division_time = int(
            self.databasehandler.divisions.iloc[self.current_division_index]["t"]
        )
        self.update_chunk_from_frame_signal.emit(division_time)

        # update the selected nodes in the TreeWidget
        label = self.databasehandler.divisions.iloc[self.current_division_index]["id"]
        self.tracks_viewer.selected_nodes._list = []
        self.tracks_viewer.selected_nodes._list.append(label)
        self.tracks_viewer.selected_nodes.list_updated.emit()

        self.update_division_counter()

    def _check_selected_node_matches_division(self):
        """Check if the selected node matches the current division label."""
        selected_nodes = self.tracks_viewer.selected_nodes._list

        if len(selected_nodes) != 1:
            self.division_counter.setStyleSheet("color: gray;")
            return

        selected_node = selected_nodes[0]
        division_ids = self.databasehandler.divisions["id"].values

        try:
            index = np.where(division_ids == selected_node)[0][0]
            self.current_division_index = index
            self.division_counter.setText(
                f"{index + 1}/{len(self.databasehandler.divisions)}"
            )
            self.division_counter.setStyleSheet("")
        except IndexError:
            self.division_counter.setStyleSheet("color: gray;")
