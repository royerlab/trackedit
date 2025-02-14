import napari
from napari.utils.notifications import show_warning
from motile_tracker.application_menus.editing_menu import EditingMenu
from motile_tracker.data_views import TracksViewer, TreeWidget   
from motile_tracker.data_model.solution_tracks import SolutionTracks

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
)
from PyQt5.QtCore import Qt
from qtpy.QtCore import Signal
from trackedit.DatabaseHandler import DatabaseHandler

class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit()

class TrackEditSidebar(QWidget):

    change_chunk = Signal(str)
    goto_frame = Signal(int)

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.tracks_viewer = TracksViewer.get_instance(viewer)

        # ===============================
        # TIME INTERACTION UI ELEMENTS
        # ===============================

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

        # ===============================
        # RED FLAG SECTION UI ELEMENTS
        # ===============================

        # Label showing which red flag is currently active (e.g., "3/80")
        self.red_flag_counter = ClickableLabel("0/0")  # Will be updated later with actual counts

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

        # ===============================

        #Define entire widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(QLabel(r"""<h2>Navigation</h2>""" ))
        main_layout.addLayout(time_input_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.chunk_label, alignment=Qt.AlignCenter)
        main_layout.addWidget(QLabel(r"""<h2>Red Flags</h2>""" ))
        main_layout.addLayout(red_flag_layout)

        self.setLayout(main_layout)
        # self.setMaximumHeight(250)

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

# overwrite EditingMenu to make sure outlines of points in TreePlot are transparent
class CustomEditingMenu(EditingMenu):
    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer)  # Call the original init method

        # Create the label
        nav_label = QLabel(r"""<h2>Edit tracks</h2>""")

        # Get the existing layout
        layout = self.layout()  # This retrieves the QVBoxLayout from EditingMenu

        # Insert the label at the beginning
        layout.insertWidget(0, nav_label)

class TrackEditClass():
    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        self.viewer = viewer

        self.TreeWidget = TreeWidget(self.viewer)
        self.TrackEditSidebar = TrackEditSidebar(self.viewer)
        self.EditingMenu = CustomEditingMenu(self.viewer)

        self.viewer.window.add_dock_widget(self.TreeWidget, area="bottom",name="TreeWidget")
        self.viewer.window.add_dock_widget(self.TrackEditSidebar, area='right', name='TrackEditSidebar')
        self.viewer.window.add_dock_widget(self.EditingMenu,area="right",name="EditingMenu")

        #Todo: provide entire DB_handler
        self.databasehandler = databasehandler
        self.TrackEditSidebar.change_chunk.connect(self.update_chunk_from_button)
        self.TrackEditSidebar.goto_frame.connect(self.update_chunk_from_frame)

        #Connect to napari's time slider (dims) event)
        self.viewer.dims.events.current_step.connect(self.on_dims_changed)

        #Connect red flag UI buttons
        self.current_red_flag_index = 0
        self.TrackEditSidebar.red_flag_prev_btn.clicked.connect(self.go_to_prev_red_flag)
        self.TrackEditSidebar.red_flag_next_btn.clicked.connect(self.go_to_next_red_flag)
        self.TrackEditSidebar.red_flag_ignore_btn.clicked.connect(self.ignore_red_flag)
        self.TrackEditSidebar.red_flag_counter.clicked.connect(self.goto_red_flag)

        self.add_tracks()
        self.TrackEditSidebar.update_chunk_label()
        self.update_red_flag_counter_and_info()


    def update_red_flag_counter_and_info(self):
        """Update the red flag label to show the current red flag index and total count."""
        total = len(self.databasehandler.red_flags)
        print(' RF: total = ', total, 'current index', self.current_red_flag_index)
        if total > 0:
            # Display indices as 1-indexed (e.g., "3/80")
            self.TrackEditSidebar.red_flag_counter.setText(f"{self.current_red_flag_index + 1}/{total}")
            df_rf = self.databasehandler.red_flags.iloc[[self.current_red_flag_index]]
            text = f"{df_rf.iloc[0].id} {df_rf.iloc[0].event} at t={df_rf.iloc[0].t}"
            self.TrackEditSidebar.red_flag_info.setText(text)
        else:
            self.TrackEditSidebar.red_flag_counter.setText("0/0")
            self.TrackEditSidebar.red_flag_info.setText("-")

    def go_to_next_red_flag(self):
        """Navigate to the next red flag in the list and jump to that timepoint."""
        total = len(self.databasehandler.red_flags)
        if total == 0:
            return
        # Increment index (wrap around if needed)
        self.current_red_flag_index = (self.current_red_flag_index + 1) % total
        self.goto_red_flag()

    def go_to_prev_red_flag(self):
        """Navigate to the previous red flag in the list and jump to that timepoint."""
        total = len(self.databasehandler.red_flags)
        if total == 0:
            return
        # Decrement index (wrap around if needed)
        self.current_red_flag_index = (self.current_red_flag_index - 1) % total
        self.goto_red_flag()

    def goto_red_flag(self):
        """Jump to the time of the current red flag."""
        red_flag_time = int(self.databasehandler.red_flags.iloc[self.current_red_flag_index]["t"])
        self.update_chunk_from_frame(red_flag_time)

        #update the selected nodes in the TreeWidget
        tv = TracksViewer.get_instance(self.viewer)
        label = self.databasehandler.red_flags.iloc[self.current_red_flag_index]["id"]
        tv.selected_nodes._list = []
        tv.selected_nodes._list.append(label)
        tv.selected_nodes.list_updated.emit()

        self.update_red_flag_counter_and_info()

    def ignore_red_flag(self):
        """Ignore the current red flag and remove it from the list."""
        id = self.databasehandler.red_flags.iloc[self.current_red_flag_index]["id"]
        self.databasehandler.seg_ignore_red_flag(id)

        if self.current_red_flag_index >= len(self.databasehandler.red_flags):
            self.current_red_flag_index = self.current_red_flag_index - 1
        self.goto_red_flag()

    def add_tracks(self):
        """Add a solution set of tracks to the tracks viewer results list

        Args:
            tracker (ultrack.Tracker): the ultrack tracker containing the solution
            name (str): the display name of the solution tracks
        """

        # create tracks object
        tracks = SolutionTracks(
            graph = self.databasehandler.nxgraph,
            segmentation = self.databasehandler.segments,
            pos_attr=("z","y", "x"),
            time_attr="t",
            scale = [1,4,1,1],
        )

        # add tracks to viewer
        tracksviewer = TracksViewer.get_instance(self.viewer)
        tracksviewer.tracks_list.add_tracks(tracks,name=self.databasehandler.name)
        self.viewer.layers.selection.active = self.viewer.layers[self.databasehandler.name+'_seg']   #select segmentation layer

        self.check_navigation_button_validity()

        #ToDo: check if all tracks are added or overwritten

    def update_chunk_from_button(self, direction: str):
        cur_chunk = self.databasehandler.time_chunk
        current_slider_position = self.viewer.dims.current_step[0]

        if direction not in ['prev', 'next']:
            raise ValueError(f"Invalid direction: {direction}")

        #change the time chunk index
        if direction == 'prev':
            new_chunk = cur_chunk - 1
        else:
            new_chunk = cur_chunk + 1

        #check if the new chunk is within the limits
        if new_chunk < 0:
            new_chunk = 0
        elif new_chunk == self.databasehandler.num_time_chunks:
            new_chunk = self.databasehandler.num_time_chunks - 1

        self.databasehandler.set_time_chunk(new_chunk)
        self.add_tracks()

        if direction == 'prev':
            desired_chunk_time = self.databasehandler.time_chunk_length - self.databasehandler.time_chunk_overlap + current_slider_position - 1
        else:
            desired_chunk_time = -self.databasehandler.time_chunk_length + self.databasehandler.time_chunk_overlap + current_slider_position + 1

        self.set_time_slider(desired_chunk_time)       
        self.TrackEditSidebar.update_chunk_label()


    def update_chunk_from_frame(self, frame: int):
        """Handle navigation by a user-entered time frame.
        
        This calculates the chunk containing the given frame.
        For example, if each chunk is 100 frames, frame 235 belongs in chunk 2.
        """
        if frame<0: 
            frame = 0
        elif frame >= self.databasehandler.Tmax:
            frame = self.databasehandler.Tmax - 1

        new_chunk = self.databasehandler.find_chunk_from_frame(frame)

        if new_chunk < 0:
            new_chunk = 0
        elif new_chunk >= self.databasehandler.num_time_chunks:
            new_chunk = self.databasehandler.num_time_chunks - 1

        cur_chunk = self.databasehandler.time_chunk

        self.databasehandler.set_time_chunk(new_chunk)
        if cur_chunk != new_chunk:
            self.add_tracks()        

        chunk_frame = frame - self.databasehandler.time_chunk_starts[self.databasehandler.time_chunk]
        self.set_time_slider(chunk_frame)
        self.TrackEditSidebar.update_chunk_label()
        self.update_time_label()

    def set_time_slider(self, chunk_frame):
        self.viewer.dims.current_step = (chunk_frame, *self.viewer.dims.current_step[1:])

    def on_dims_changed(self, _):
        self.update_time_label()

    def update_time_label(self):
        chunk = self.databasehandler.time_chunk
        cur_frame = self.viewer.dims.current_step[0]
        cur_world_time = cur_frame + self.databasehandler.time_chunk_starts[chunk]

        self.TrackEditSidebar.time_input.setText(str(cur_world_time))

    def check_navigation_button_validity(self):
        #enable/disable buttons if on first/last chunk
        chunk = self.databasehandler.time_chunk
        if chunk == 0:
            self.TrackEditSidebar.time_prev_btn.setEnabled(False)
        else:
            self.TrackEditSidebar.time_prev_btn.setEnabled(True)

        if chunk == self.databasehandler.num_time_chunks - 1:
            self.TrackEditSidebar.time_next_btn.setEnabled(False)
        else:
            self.TrackEditSidebar.time_next_btn.setEnabled(True)