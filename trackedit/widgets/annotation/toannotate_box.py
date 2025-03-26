from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QHBoxLayout, QPushButton

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
        self.label_int_mapping = {
            v["name"]: k for k, v in self.databasehandler.label_mapping_dict.items()
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

        # Change to horizontal layout for H,S,M buttons
        annotation_type_layout = QHBoxLayout()

        self.hair_btn = QPushButton("H")
        self.support_btn = QPushButton("S")
        self.mantle_btn = QPushButton("M")

        annotation_type_layout.addWidget(self.hair_btn)
        annotation_type_layout.addWidget(self.support_btn)
        annotation_type_layout.addWidget(self.mantle_btn)

        self.layout.addLayout(annotation_type_layout)

        # Connect buttons
        self.toannotate_prev_btn.clicked.connect(self.go_to_prev_annotation)
        self.toannotate_next_btn.clicked.connect(self.go_to_next_annotation)
        self.toannotate_counter.clicked.connect(self.goto_annotation)
        self.hair_btn.clicked.connect(self.on_hair_clicked)
        self.support_btn.clicked.connect(self.on_support_clicked)
        self.mantle_btn.clicked.connect(self.on_mantle_clicked)
        self.tracks_viewer.tracks_updated.connect(self.update_toannotate)
        self.tracks_viewer.selected_nodes.list_updated.connect(self._update_upon_click)

    def update_toannotate(self):
        """Update the annotations and the annotation counter"""
        self.databasehandler.recompute_toannotate()
        self.update_annotation_counter()

    def update_annotation_counter(self):
        """Update the annotation counter to show the current annotation index and total count."""
        total = len(self.databasehandler.toannotate)
        if total > 0:
            self.toannotate_counter.setText(
                f"{self.current_annotation_index + 1}/{total}"
            )
        else:
            self.toannotate_counter.setText("0/0")

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

    def on_hair_clicked(self):
        self.annotate_track("hair")

    def on_support_clicked(self):
        self.annotate_track("support")

    def on_mantle_clicked(self):
        self.annotate_track("mantle")

    def annotate_track(self, label_str: str):
        label_int = self.get_label_int(label_str)

        track_id = self.databasehandler.df_full.loc[self.current_selected_cell]["track_id"]

        self.databasehandler.annotate_track(
            track_id, label_int
        )  # make changes in the database
        self.update_toannotate()  # update the toannotate list in databasehandler
        self._update_upon_click()  # update the counter and buttons
        self.refresh_annotation_layer.emit()  # refresh the annotation layer

    def get_label_int(self, label_str: str):
        # Use inverse of databasehandler's label_mapping
        try:
            return self.label_int_mapping[label_str]
        except KeyError:
            raise ValueError(f"Invalid label: {label_str}")

    def toggle_buttons(self, toggle: bool, label_int: int | None):
        if toggle:
            self.toannotate_counter.setStyleSheet("")
        else:
            self.toannotate_counter.setStyleSheet("color: gray;")

        self.toannotate_prev_btn.setEnabled(toggle)
        self.toannotate_next_btn.setEnabled(toggle)

        # if label_int is not None, make the corresponding button yellow:
        if label_int != -1:
            # Reset all buttons to default style first
            self.hair_btn.setStyleSheet("")
            self.support_btn.setStyleSheet("")
            self.mantle_btn.setStyleSheet("")
            opacity_factor = 0.75
            # Then set only the matching button to yellow
            if label_int == self.label_int_mapping["hair"]:
                color = self.databasehandler.label_mapping_dict[label_int]["color"]
                self.hair_btn.setStyleSheet(
                    f"background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, "
                    f"{opacity_factor});"
                )
            elif label_int == self.label_int_mapping["support"]:
                color = self.databasehandler.label_mapping_dict[label_int]["color"]
                self.support_btn.setStyleSheet(
                    f"background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, "
                    f"{opacity_factor});"
                )
            elif label_int == self.label_int_mapping["mantle"]:
                color = self.databasehandler.label_mapping_dict[label_int]["color"]
                self.mantle_btn.setStyleSheet(
                    f"background-color: rgba({int(color[0]*255)}, {int(color[1]*255)}, {int(color[2]*255)}, "
                    f"{opacity_factor});"
                )
        else:
            # Reset all buttons to default style
            self.hair_btn.setStyleSheet("")
            self.support_btn.setStyleSheet("")
            self.mantle_btn.setStyleSheet("")

    def _update_upon_click(self):
        """
        This function is called when the user clicks on a cell in the tree widget.
        It: - updates the label with the selected cell information
            - updates the counter and buttons
        """
        selected_nodes = self.tracks_viewer.selected_nodes._list

        if len(selected_nodes) != 1:
            self.selected_cell_label.setText("No cell selected")
            return

        selected_node = selected_nodes[0]
        self.current_selected_cell = selected_node

        label_int = self._update_label(selected_node)

        # update counter and buttons
        try:
            track_id = self.databasehandler.df_full.loc[self.current_selected_cell]["track_id"]
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
        cell_info = self.databasehandler.df_full[self.databasehandler.df_full["id"] == selected_node].iloc[0]
        generic_value = cell_info["generic"]  # type(generic_value) = int
        label = self.databasehandler.label_mapping(generic_value)  # type(label) = str
        self.selected_cell_label.setText(f"Selected cell: {selected_node} ({label})")

        return generic_value
