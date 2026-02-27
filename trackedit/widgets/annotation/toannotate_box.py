from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
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
        self.current_toannotate_segment = None

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

        # Create time range input fields
        self.t_begin_label = QLabel("t_begin:")
        self.t_begin_input = QLineEdit()
        self.t_begin_input.setFixedWidth(60)
        self.t_begin_input.setPlaceholderText("0")

        self.t_end_label = QLabel("t_end:")
        self.t_end_input = QLineEdit()
        self.t_end_input.setFixedWidth(60)
        self.t_end_input.setPlaceholderText("100")

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

        # Add time range input layout
        time_range_layout = QHBoxLayout()
        time_range_layout.addWidget(self.t_begin_label)
        time_range_layout.addWidget(self.t_begin_input)
        time_range_layout.addWidget(self.t_end_label)
        time_range_layout.addWidget(self.t_end_input)
        time_range_layout.addStretch()  # Add stretch to push fields to the left

        self.layout.addLayout(time_range_layout)

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
        """Jump to the time of the current annotation segment."""
        # Check if there are any annotations to navigate to
        if len(self.databasehandler.toannotate) == 0:
            return

        current_segment = self.databasehandler.toannotate.iloc[
            self.current_annotation_index
        ]
        self.current_toannotate_segment = current_segment
        annotation_time = current_segment["first_t"]
        cell_id = current_segment["first_id"]

        self.current_selected_cell = cell_id

        self.update_chunk_from_frame_signal.emit(annotation_time)

        # Update the selected nodes in the TreeWidget
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

        # Get time range from input fields
        t_begin = None
        t_end = None

        try:
            if self.t_begin_input.text().strip():
                t_begin = int(self.t_begin_input.text().strip())
            if self.t_end_input.text().strip():
                t_end = int(self.t_end_input.text().strip())
        except ValueError:
            # If time range is invalid, fall back to full track annotation
            print("Warning: Invalid time range, annotating entire track")
            t_begin = None
            t_end = None

        # Validate time range
        if t_begin is not None and t_end is not None and t_begin > t_end:
            print("Warning: t_begin > t_end, swapping values")
            t_begin, t_end = t_end, t_begin

        # Validate time range against current segment bounds (only when the selected
        # cell is inside an unannotated segment; skip for already-annotated cells)
        if self.current_toannotate_segment is not None:
            segment_first_t = self.current_toannotate_segment["first_t"]
            segment_last_t = self.current_toannotate_segment["last_t"]

            if t_begin is not None and t_begin < segment_first_t:
                print(
                    f"Warning: t_begin ({t_begin}) < segment start ({segment_first_t}), using segment start"
                )
                t_begin = segment_first_t
            if t_end is not None and t_end > segment_last_t:
                print(
                    f"Warning: t_end ({t_end}) > segment end ({segment_last_t}), using segment end"
                )
                t_end = segment_last_t

        self.databasehandler.annotate_track(
            track_id, label_int, t_begin, t_end
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
            - populates the time range fields
        """
        selected_nodes = self.tracks_viewer.selected_nodes._list

        if len(selected_nodes) != 1:
            self.selected_cell_label.setText("No cell selected")
            self.toggle_buttons(False)
            self.clear_time_range_fields()
            return

        selected_node = selected_nodes[0]
        self.current_selected_cell = selected_node

        label_int = self._update_label(selected_node)

        # update counter and buttons
        try:
            track_id = self.databasehandler.df_full.loc[self.current_selected_cell][
                "track_id"
            ]
            cell_time = self.databasehandler.df_full.loc[self.current_selected_cell][
                "t"
            ]

            # Find the segment that contains this cell (by track_id and time range)
            matching_segments = self.databasehandler.toannotate[
                (self.databasehandler.toannotate["track_id"] == track_id)
                & (self.databasehandler.toannotate["first_t"] <= cell_time)
                & (self.databasehandler.toannotate["last_t"] >= cell_time)
            ]

            if not matching_segments.empty:
                # Found the segment - update counter and enable buttons
                index = matching_segments.index[0]
                self.current_annotation_index = index
                self.current_toannotate_segment = matching_segments.iloc[0]
                self.toannotate_counter.setText(
                    f"{index + 1}/{len(self.databasehandler.toannotate)}"
                )
                self.toggle_buttons(True, label_int)

                # Populate time range fields with current segment info
                self.populate_time_range_fields(index)
            else:
                # Cell not in any unannotated segment - disable counter and buttons
                self.current_toannotate_segment = None
                self.toggle_buttons(False, label_int)
                # Show the bounds of the annotated segment containing this cell
                self.populate_annotated_segment_range(track_id, cell_time)

        except IndexError:
            # Track not found in annotations - disable counter and buttons
            self.toggle_buttons(False, label_int)
            self.clear_time_range_fields()

    def _get_label_segments(self, track_data):
        """Return list of (first_t, last_t, label) for each contiguous same-label run."""
        if track_data.empty:
            return []
        sorted_data = track_data.sort_values("t")
        times = sorted_data["t"].to_numpy()
        labels = sorted_data["generic"].to_numpy()

        segments = []
        seg_start = times[0]
        seg_label = labels[0]
        for i in range(1, len(times)):
            if labels[i] != seg_label or times[i] != times[i - 1] + 1:
                segments.append((int(seg_start), int(times[i - 1]), int(seg_label)))
                seg_start = times[i]
                seg_label = labels[i]
        segments.append((int(seg_start), int(times[-1]), int(seg_label)))
        return segments

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

        # Get track_id for additional context
        track_id = cell_info["track_id"]

        # Check annotation status of the entire track for the status line
        track_data = self.databasehandler.df_full[
            self.databasehandler.df_full["track_id"] == track_id
        ]
        unannotated_frames = track_data[
            track_data["generic"] == NodeDB.generic.default.arg
        ]

        if len(unannotated_frames) == 0:
            # Entire track is annotated
            unique_labels = track_data["generic"].unique()
            label_names = ", ".join(
                self.databasehandler.annotation_mapping(int(l))[0].upper()
                for l in unique_labels
            )
            status_line = f"entire track annotated ({label_names})"
        elif len(unannotated_frames) == len(track_data):
            # Entire track is unannotated - add time range
            first_t = track_data["t"].min()
            last_t = track_data["t"].max()
            status_line = f"entire track not annotated (t={int(first_t)}:{int(last_t)})"
        else:
            # Track is partially annotated - find annotated segments per label
            default_label = NodeDB.generic.default.arg
            annotated_segments = [
                f"{f}:{l}"
                for f, l, lbl in self._get_label_segments(track_data)
                if lbl != default_label
            ]
            status_line = (
                f"track partially annotated (t={', '.join(annotated_segments)})"
            )

        # Create two-line display
        display_text = f"Selected cell: {selected_node} ({label})\n{status_line}"
        self.selected_cell_label.setText(display_text)

        return generic_value

    def populate_time_range_fields(self, segment_index):
        """Populate the time range fields with the current segment's time range."""
        if segment_index < len(self.databasehandler.toannotate):
            segment = self.databasehandler.toannotate.iloc[segment_index]
            self.t_begin_input.setText(str(segment["first_t"]))
            self.t_end_input.setText(str(segment["last_t"]))

    def clear_time_range_fields(self):
        """Clear the time range fields."""
        self.t_begin_input.clear()
        self.t_end_input.clear()

    def populate_annotated_segment_range(self, track_id, cell_time):
        """Populate time range fields with the bounds of the contiguous same-label
        segment in track_id that contains cell_time."""
        track_data = self.databasehandler.df_full[
            self.databasehandler.df_full["track_id"] == track_id
        ]
        for first_t, last_t, _ in self._get_label_segments(track_data):
            if first_t <= cell_time <= last_t:
                self.t_begin_input.setText(str(first_t))
                self.t_end_input.setText(str(last_t))
                return

    def populate_full_track_range(self, track_id):
        """Populate the time range fields with the full track range for context."""
        track_data = self.databasehandler.df_full[
            self.databasehandler.df_full["track_id"] == track_id
        ]
        if not track_data.empty:
            first_t = track_data["t"].min()
            last_t = track_data["t"].max()
            self.t_begin_input.setText(str(int(first_t)))
            self.t_end_input.setText(str(int(last_t)))
