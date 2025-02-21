import napari
import numpy as np
from napari.utils.notifications import show_warning
from motile_tracker.data_views import TracksViewer   
from trackedit.DatabaseHandler import DatabaseHandler
from trackedit.widgets.ClickableLabel import ClickableLabel
from qtpy.QtCore import Signal, Qt
from qtpy.QtWidgets import (
    QWidget, 
    QVBoxLayout, 
    QPushButton, 
    QGroupBox, 
    QHBoxLayout, 
    QLabel,   
    QLineEdit
)

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

        # ===============================
        # TIME INTERACTION UI ELEMENTS
        # ===============================

        time_box = QGroupBox()
        time_box_layout = QVBoxLayout()

        #Define the buttons
        self.time_prev_btn = QPushButton("prev (<)")
        self.time_prev_btn.clicked.connect(self.press_prev)
        self.time_next_btn = QPushButton("next (>)")
        self.time_next_btn.clicked.connect(self.press_next)
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.time_prev_btn)
        button_layout.addWidget(self.time_next_btn)

        #Define the time window label
        self.chunk_label = QLabel("temp. label")

        # Define an input field that shows the current time frame
        # and allows the user to type a new frame number.
        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("Enter time")
        self.time_input.returnPressed.connect(self.on_time_input_entered)

        # Create a horizontal layout for the label and input field.
        time_input_layout = QHBoxLayout()
        time_input_label = QLabel("time = ")  # This is the text before the input field.
        time_input_layout.addWidget(time_input_label)
        time_input_layout.addWidget(self.time_input)

        time_box_layout.addWidget(QLabel(r"""<h3>Time navigation</h3>""" ))
        time_box_layout.addLayout(time_input_layout)
        time_box_layout.addLayout(button_layout)
        time_box_layout.addWidget(self.chunk_label, alignment=Qt.AlignCenter)
        time_box.setLayout(time_box_layout)
        time_box.setMaximumHeight(155)

        # ===============================
        # RED FLAG SECTION UI ELEMENTS
        # ===============================

        #The widget that contains the red flag controls
        redflag_box = QGroupBox()
        redflag_box_layout = QVBoxLayout()

        # Label showing which red flag is currently active (e.g., "3/80")
        self.red_flag_counter = ClickableLabel("0/0")  # Will be updated later with actual counts
        self.red_flag_counter.setFixedWidth(50)  

        # Button to go to the previous red flag ("<")
        self.red_flag_prev_btn = QPushButton("<")
        self.red_flag_prev_btn.setFixedWidth(30)  

        # Button to go to the next red flag (">")
        self.red_flag_next_btn = QPushButton(">")
        self.red_flag_next_btn.setFixedWidth(30)  

        # Button to ignore the current red flag ("ignore")
        self.red_flag_ignore_btn = QPushButton("ignore")
        self.red_flag_info = QLabel("info")

        # Create a horizontal layout to contain the red flag controls
        red_flag_layout = QVBoxLayout()
        red_flag_layout_row1 = QHBoxLayout()

        red_flag_layout_row1.addWidget(self.red_flag_prev_btn)
        red_flag_layout_row1.addWidget(self.red_flag_counter)
        red_flag_layout_row1.addWidget(self.red_flag_next_btn)
        red_flag_layout_row1.addWidget(self.red_flag_ignore_btn)

        red_flag_layout.addLayout(red_flag_layout_row1)
        red_flag_layout.addWidget(self.red_flag_info)

        redflag_box_layout.addWidget(QLabel(r"""<h3>Red Flags</h3>""" ))
        redflag_box_layout.addLayout(red_flag_layout)
        redflag_box.setLayout(redflag_box_layout)
        redflag_box.setMaximumHeight(125)

        #Connect red flag UI buttons
        self.current_red_flag_index = 0
        self.red_flag_prev_btn.clicked.connect(self.go_to_prev_red_flag)
        self.red_flag_next_btn.clicked.connect(self.go_to_next_red_flag)
        self.red_flag_ignore_btn.clicked.connect(self.ignore_red_flag)
        self.red_flag_counter.clicked.connect(self.goto_red_flag)
        self.tracks_viewer.tracks_updated.connect(self.update_red_flags)


        # ===============================
        # Division clicker
        # ===============================

        #The widget that contains the division controls
        division_box = QGroupBox()
        division_box_layout = QVBoxLayout()

        # Label showing which division is currently active (e.g., "3/80")
        self.division_counter = ClickableLabel("0/0")  # Will be updated later with actual counts
        self.division_counter.setFixedWidth(50)  

        # Button to go to the previous division ("<")
        self.division_prev_btn = QPushButton("<")
        self.division_prev_btn.setFixedWidth(30)  

        # Button to go to the next division (">")
        self.division_next_btn = QPushButton(">")
        self.division_next_btn.setFixedWidth(30)  

        # Create a horizontal layout to contain the division controls
        division_layout = QHBoxLayout()
        division_layout.addWidget(self.division_prev_btn)
        division_layout.addWidget(self.division_counter)
        division_layout.addWidget(self.division_next_btn)

        division_box_layout.addWidget(QLabel(r"""<h3>Divisions</h3>""" ), alignment=Qt.AlignLeft)
        division_box_layout.addLayout(division_layout)
        division_box_layout.setAlignment(division_layout, Qt.AlignLeft)
        division_box.setLayout(division_box_layout)
        division_box.setMaximumHeight(100)

        #Connect division UI buttons
        self.current_division_index = 0
        self.division_prev_btn.clicked.connect(self.go_to_prev_division)
        self.division_next_btn.clicked.connect(self.go_to_next_division)
        self.division_counter.clicked.connect(self.goto_division)
        self.tracks_viewer.tracks_updated.connect(self.update_divisions)

        # ===============================

        #Define entire widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(time_box)
        main_layout.addWidget(redflag_box)
        main_layout.addWidget(division_box)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(10, 2, 10, 2)
        self.setLayout(main_layout)

        # Connect division UI buttons
        self.division_prev_btn.clicked.connect(self.go_to_prev_division)
        self.division_next_btn.clicked.connect(self.go_to_next_division)
        self.division_counter.clicked.connect(self.goto_division)
        self.tracks_viewer.tracks_updated.connect(self.update_divisions)
        self.tracks_viewer.selected_nodes.list_updated.connect(self._check_selected_node_matches_division)
        self.tracks_viewer.selected_nodes.list_updated.connect(self._check_selected_node_matches_red_flag)
        
        #Connect to napari's time slider (dims) event)
        self.viewer.dims.events.current_step.connect(self.on_dims_changed)
    #===============================================
    # Navigation
    #===============================================

    def set_time_slider(self, chunk_frame):
        self.viewer.dims.current_step = (chunk_frame, *self.viewer.dims.current_step[1:])

    def on_dims_changed(self, _) -> None:
        """Called when the time slider is moved by the user, to update the "world time" label."""
        self.update_time_label()

    def update_time_label(self) -> None:
        """Update the time label to show the current time in the world."""
        chunk = self.databasehandler.time_chunk
        cur_frame = self.viewer.dims.current_step[0]
        cur_world_time = cur_frame + self.databasehandler.time_chunk_starts[chunk]
        self.time_input.setText(str(cur_world_time))

    def check_navigation_button_validity(self) -> None:
        """Check if the navigation chunk buttons should be enabled or disabled based on the current chunk. Enable/disable buttons if on first/last chunk."""
        chunk = self.databasehandler.time_chunk
        if chunk == 0:
            self.time_prev_btn.setEnabled(False)
        else:
            self.time_prev_btn.setEnabled(True)

        if chunk == self.databasehandler.num_time_chunks - 1:
            self.time_next_btn.setEnabled(False)
        else:
            self.time_next_btn.setEnabled(True)

    def press_prev(self):
        self.change_chunk.emit('prev')

    def press_next(self):
        self.change_chunk.emit('next')

    def on_time_input_entered(self):
        """Called when the user presses Enter in the time_input field."""
        try:
            frame = int(self.time_input.text())
            self.goto_frame.emit(frame)
        except ValueError:
            # If the user entered a non-integer value, show a warning and do nothing.
            cur_frame = self.tracks_viewer.tracks.segmentation.current_time
            self.goto_frame.emit(cur_frame)
            show_warning("Time invalid, nothing changed.")
        pass

    def update_chunk_label(self):
        time_window = self.tracks_viewer.tracks.segmentation.time_window
        label = f" time window [{time_window[0]} : {time_window[1]-1}]"
        self.chunk_label.setText(label)

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
            self.red_flag_counter.setStyleSheet("color: gray;")
            return

        selected_node = selected_nodes[0]
        red_flag_ids = self.databasehandler.red_flags['id'].values
        
        try:
            index = np.where(red_flag_ids == selected_node)[0][0]
            # Found the node in red flags - update counter and remove grey
            self.current_red_flag_index = index
            self.red_flag_counter.setText(f"{index + 1}/{len(self.databasehandler.red_flags)}")
            self.red_flag_counter.setStyleSheet("")
        except IndexError:
            # Node not found in red flags - grey out counter
            self.red_flag_counter.setStyleSheet("color: gray;")

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
            self.division_counter.setStyleSheet("color: gray;")
            return

        selected_node = selected_nodes[0]
        division_ids = self.databasehandler.divisions['id'].values
        
        try:
            index = np.where(division_ids == selected_node)[0][0]
            self.current_division_index = index
            self.division_counter.setText(f"{index + 1}/{len(self.databasehandler.divisions)}")
            self.division_counter.setStyleSheet("")
        except IndexError:
            self.division_counter.setStyleSheet("color: gray;")