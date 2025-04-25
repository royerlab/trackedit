from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QPushButton
from ultrack.core.database import NodeDB

from trackedit.widgets.base_box import NavigationBox
from trackedit.widgets.ClickableLabel import ClickableLabel


class ToAnnotateBox(NavigationBox):

    update_chunk_from_frame_signal = Signal(int)
    refresh_annotation_layer = Signal()

    def __init__(self, tracks_viewer, databasehandler):
        super().__init__("Annotations", max_height=200)
        self.setObjectName("ToAnnotateBox")
        self.tracks_viewer = tracks_viewer
        self.databasehandler = databasehandler
        self.current_annotation_index = 0
        self.current_selected_cell = None

        # Create reverse mapping using the "name" field
        self.annotation_int_mapping = {
            v["name"]: k
            for k, v in self.databasehandler.annotation_mapping_dict.items()
        }

        # Create selected cell info label
        self.selected_cell_label = ClickableLabel("No cell selected")

        # Create controls
        self.toannotate_counter = ClickableLabel("0/0")
        self.toannotate_counter.setFixedWidth(80)

        self.toannotate_prev_btn = QPushButton("<")
        self.toannotate_prev_btn.setFixedWidth(30)

        self.toannotate_next_btn = QPushButton(">")
        self.toannotate_next_btn.setFixedWidth(30)

        # Layout
        selected_cell_layout = QHBoxLayout()
        selected_cell_layout.addWidget(self.selected_cell_label)
        self.layout.addLayout(selected_cell_layout)

        toannotate_layout = QHBoxLayout()
        toannotate_layout.addWidget(self.toannotate_prev_btn)
        toannotate_layout.addWidget(self.toannotate_counter)
        toannotate_layout.addWidget(self.toannotate_next_btn)

        self.layout.addLayout(toannotate_layout)
        self.layout.setAlignment(toannotate_layout, Qt.AlignLeft)

        # Change to grid layout for annotation buttons
        annotation_type_layout = QGridLayout()

        self.label_buttons = {}  # Dictionary to store buttons by label name
        row = 0
        col = 0
        for (
            label_id,
            label_info,
        ) in self.databasehandler.annotation_mapping_dict.items():
            if label_id != NodeDB.generic.default.arg:  # Skip the "none" label
                btn = QPushButton(
                    label_info["name"][0].upper()
                )  # First letter as button text
                self.label_buttons[label_info["name"]] = btn
                annotation_type_layout.addWidget(btn, row, col)
                col += 1
                if col >= 4:  # Start a new row after 4 buttons
                    col = 0
                    row += 1

        self.layout.addLayout(annotation_type_layout)
        self._update_upon_click()

        # Connect buttons
        self.toannotate_prev_btn.clicked.connect(self.go_to_prev_annotation)
        self.toannotate_next_btn.clicked.connect(self.go_to_next_annotation)
        self.toannotate_counter.clicked.connect(self.goto_annotation)
        for label_name, btn in self.label_buttons.items():
            btn.clicked.connect(
                lambda checked, ln=label_name: self.on_label_clicked(ln)
            )
        self.tracks_viewer.tracks_updated.connect(self.update_toannotate)
        self.tracks_viewer.selected_nodes.list_updated.connect(self._update_upon_click)

    def update_toannotate(self):
        """Update the annotations and the annotation counter"""
        self.databasehandler.recompute_toannotate()
        self.update_annotation_counter()

    def update_annotation_counter(self):
        """Update the annotation counter to show the current annotation index and total count."""
        total = len(self.databasehandler.toannotate)

        # Update current_annotation_index based on total
        if total == 0:
            self.current_annotation_index = 0
            self.toannotate_counter.setText("0/0")
        else:
            # Keep same index unless we're beyond the end
            self.current_annotation_index = min(
                self.current_annotation_index, total - 1
            )
            self.toannotate_counter.setText(
                f"{self.current_annotation_index + 1}/{total}"
            )

    def go_to_next_annotation(self):
        """Navigate to the next annotation in the list and jump to that timepoint."""
        total = len(self.databasehandler.toannotate)
        if total == 0:
            return
        self.current_annotation_index = (self.current_annotation_index + 1) % total
        self.goto_annotation()

    def go_to_prev_annotation(self):
        """Navigate to the previous annotation in the list and jump to that timepoint."""
        total = len(self.databasehandler.toannotate)
        if total == 0:
            return
        self.current_annotation_index = (self.current_annotation_index - 1) % total
        self.goto_annotation()

    def goto_annotation(self):
        """Jump to the time of the current annotation."""
        # Check if there are any annotations to navigate to
        if len(self.databasehandler.toannotate) == 0:
            return

        annotation_time = self.databasehandler.toannotate.iloc[
            self.current_annotation_index
        ]["first_t"]
        cell_id = self.databasehandler.toannotate.iloc[self.current_annotation_index][
            "first_id"
        ]

        self.current_selected_cell = cell_id

        self.update_chunk_from_frame_signal.emit(annotation_time)

        # #update the selected nodes in the TreeWidget
        self.tracks_viewer.selected_nodes._list = []
        self.tracks_viewer.selected_nodes._list.append(cell_id)
        self.tracks_viewer.selected_nodes.list_updated.emit()

        self.update_annotation_counter()

    def on_label_clicked(self, label_name: str):
        """Generic handler for any label button click"""
        self.annotate_track(label_name)

    def annotate_track(self, label_str: str):
        label_int = self.get_label_int(label_str)

        track_id = self.databasehandler.df_full.loc[self.current_selected_cell][
            "track_id"
        ]

        self.databasehandler.annotate_track(
            track_id, label_int
        )  # make changes in the database
        self.update_toannotate()  # update the toannotate list in databasehandler
        self._update_upon_click()  # update the counter and buttons
        self.refresh_annotation_layer.emit()  # refresh the annotation layer

    def get_label_int(self, label_str: str):
        # Use inverse of databasehandler's annotation_mapping
        try:
            return self.annotation_int_mapping[label_str]
        except KeyError:
            raise ValueError(f"Invalid label: {label_str}")

    def toggle_buttons(self, toggle: bool, label_int=None):
        """Toggle button colors based on track annotation and update counter style"""
        # Reset all buttons to default style
        for btn in self.label_buttons.values():
            btn.setStyleSheet("")

        if toggle:
            self.toannotate_counter.setStyleSheet("")
        else:
            self.toannotate_counter.setStyleSheet("color: gray;")

        self.toannotate_prev_btn.setEnabled(toggle)
        self.toannotate_next_btn.setEnabled(toggle)

        # Handle button highlighting based on label
        if label_int is not None:
            opacity_factor = 0.75
            # Find the button corresponding to this label and highlight it
            for (
                label_id,
                label_info,
            ) in self.databasehandler.annotation_mapping_dict.items():
                if label_id == label_int:
                    label_name = label_info["name"]
                    if label_name in self.label_buttons:
                        color = label_info["color"]
                        self.label_buttons[label_name].setStyleSheet(
                            f"background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, "
                            f"{int(color[2]*255)}, {opacity_factor});"
                        )

    def _update_upon_click(self):
        """
        This function is called when the user clicks on a cell in the tree widget.
        It: - updates the label with the selected cell information
            - updates the counter and buttons
        """
        selected_nodes = self.tracks_viewer.selected_nodes._list

        if len(selected_nodes) != 1:
            self.selected_cell_label.setText("No cell selected")
            self.toggle_buttons(False)
            return

        selected_node = selected_nodes[0]
        self.current_selected_cell = selected_node

        label_int = self._update_label(selected_node)

        # update counter and buttons
        try:
            track_id = self.databasehandler.df_full.loc[self.current_selected_cell][
                "track_id"
            ]
            index = self.databasehandler.toannotate.index[
                self.databasehandler.toannotate["track_id"] == track_id
            ][0]
            # Found the track in annotations - update counter and remove grey
            self.current_annotation_index = index
            self.toannotate_counter.setText(
                f"{index + 1}/{len(self.databasehandler.toannotate)}"
            )
            self.toggle_buttons(True, label_int)
        except IndexError:
            # Track not found in annotations - disable counter and buttons
            self.toggle_buttons(False, label_int)

    def _update_label(self, selected_node) -> int:
        """
        This function is called when the user clicks on a cell in the tree widget.
        It: - updates the label with the selected cell information
        """
        # update label
        cell_info = self.databasehandler.df_full[
            self.databasehandler.df_full["id"] == selected_node
        ].iloc[0]
        generic_value = cell_info["generic"]  # type(generic_value) = int
        label = self.databasehandler.annotation_mapping(
            generic_value
        )  # type(label) = str
        self.selected_cell_label.setText(f"Selected cell: {selected_node} ({label})")

        return generic_value
