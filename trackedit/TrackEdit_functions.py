import napari
import numpy as np
from napari.utils.notifications import show_warning
from motile_tracker.application_menus.editing_menu import EditingMenu
from motile_tracker.data_views import TracksViewer, TreeWidget   
from motile_tracker.data_model.solution_tracks import SolutionTracks
from motile_toolbox.candidate_graph import NodeAttr
from ultrack.core.database import NodeDB, get_node_values
from trackedit.hierarchy_viz_widget import HierarchyVizWidget

from qtpy.QtWidgets import (
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
    QLabel,
    QLineEdit,
    QTabWidget,
    QGroupBox,
)
from PyQt5.QtGui import QIntValidator, QValidator

from PyQt5.QtCore import Qt
from qtpy.QtCore import Signal
from trackedit.DatabaseHandler import DatabaseHandler

class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit()

class NavigationWidget(QWidget):

    change_chunk = Signal(str)
    goto_frame = Signal(int)

    def __init__(self, viewer: napari.Viewer):
        super().__init__()

        self.tracks_viewer = TracksViewer.get_instance(viewer)

        # ===============================
        # TIME INTERACTION UI ELEMENTS
        # ===============================

        time_box = QGroupBox("Time")
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

        time_box_layout.addWidget(QLabel(r"""<h2>Time navigation</h2>""" ))
        time_box_layout.addLayout(time_input_layout)
        time_box_layout.addLayout(button_layout)
        time_box_layout.addWidget(self.chunk_label, alignment=Qt.AlignCenter)
        time_box.setLayout(time_box_layout)


        # ===============================
        # RED FLAG SECTION UI ELEMENTS
        # ===============================

        redflag_box = QGroupBox("Time")
        redflag_box_layout = QVBoxLayout()

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

        redflag_box_layout.addWidget(QLabel(r"""<h2>Red Flags</h2>""" ))
        redflag_box_layout.addLayout(red_flag_layout)
        redflag_box.setLayout(redflag_box_layout)
        # ===============================

        #Define entire widget
        main_layout = QVBoxLayout()
        main_layout.addWidget(time_box)
        main_layout.addWidget(redflag_box)
        # main_layout.addLayout(time_input_layout)
        # main_layout.addLayout(button_layout)
        # main_layout.addWidget(self.chunk_label, alignment=Qt.AlignCenter)

        self.setLayout(main_layout)
        # self.setMaximumWidth(300)
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

class CustomEditingMenu(EditingMenu):

    add_cell_button_pressed = Signal(int)

    def __init__(self, viewer: napari.Viewer):
        super().__init__(viewer)  # Call the original init method

        main_layout = self.layout()  # This retrieves the QVBoxLayout from EditingMenu

        # Create the label and insert in beginning
        label = QLabel(r"""<h2>Edit tracks</h2>""")
        main_layout.insertWidget(0, label)

        #add cell row
        add_cell_layout = QHBoxLayout()                     
        self.add_cell_btn = QPushButton("Add cell")        
        self.add_cell_btn.setEnabled(False)       
        self.add_cell_btn.clicked.connect(self.add_cell_from_button)            
        add_cell_layout.addWidget(self.add_cell_btn)          
        self.cell_input = QLineEdit()                       
        self.cell_input.setValidator(QIntValidator())
        self.cell_input.textChanged.connect(self.update_add_cell_btn_state)  
        add_cell_layout.addWidget(self.cell_input)      

        #duplicate cell row
        duplicate_cell_layout = QHBoxLayout()                
        self.duplicate_cell_btn = QPushButton("dupl.") 
        self.duplicate_cell_btn.setEnabled(False)            
        duplicate_cell_layout.addWidget(self.duplicate_cell_btn)  
        self.duplicate_cell_id_input = QLineEdit()            
        self.duplicate_cell_id_input.setValidator(QIntValidator()) 
        duplicate_cell_layout.addWidget(self.duplicate_cell_id_input) 
        duplicate_cell_layout.addWidget(QLabel("to t="))      
        self.duplicate_time_input = QLineEdit()              
        self.duplicate_time_input.setValidator(QIntValidator())  
        self.duplicate_time_input.setFixedWidth(40)           
        duplicate_cell_layout.addWidget(self.duplicate_time_input)  

        self.duplicate_cell_id_input.textChanged.connect(self.update_duplicate_cell_btn_state)  
        self.duplicate_time_input.textChanged.connect(self.update_duplicate_cell_btn_state)    

        # Retrieve the node_box widget from the layout.
        # Since we've inserted a label at index 0, the node_box is now at index 1.
        node_box = main_layout.itemAt(1).widget()             
        node_box.layout().addLayout(add_cell_layout)          
        node_box.layout().addLayout(duplicate_cell_layout)   
        node_box.setMaximumHeight(150)

        self.setMaximumHeight(430)

    def update_add_cell_btn_state(self, text):                
        state, _, _ = self.cell_input.validator().validate(text, 0)  
        self.add_cell_btn.setEnabled(state == QValidator.Acceptable)   

    def update_duplicate_cell_btn_state(self, text):        
        state1, _, _ = self.duplicate_cell_id_input.validator().validate(
            self.duplicate_cell_id_input.text(), 0)        
        state2, _, _ = self.duplicate_time_input.validator().validate(
            self.duplicate_time_input.text(), 0)            
        self.duplicate_cell_btn.setEnabled(
            state1 == QValidator.Acceptable and state2 == QValidator.Acceptable) 
        
    def add_cell_from_button(self):
        node_id = int(self.cell_input.text())
        self.add_cell_button_pressed.emit(node_id)

class TrackEditClass():
    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        self.viewer = viewer
        self.databasehandler = databasehandler

        self.TreeWidget = TreeWidget(self.viewer)
        self.NavigationWidget = NavigationWidget(self.viewer)
        self.EditingMenu = CustomEditingMenu(self.viewer)


        tabwidget_right = QTabWidget()
        tabwidget_right.addTab(self.NavigationWidget, "Navigation")
        tabwidget_right.addTab(self.EditingMenu, "Edit Tracks")
        tabwidget_right.setMaximumWidth(330) 
        self.viewer.window.add_dock_widget(tabwidget_right, area='right', name='TrackEdit Widgets')

        self.hier_widget = HierarchyVizWidget(viewer = viewer,scale=self.databasehandler.scale, config = self.databasehandler.config_adjusted)
        hier_shape = self.hier_widget.ultrack_array.shape
        tmax = self.databasehandler.data_shape_chunk[0]
        self.hier_widget.ultrack_array.shape = (tmax, *hier_shape[1:])
        # self.viewer.window.add_dock_widget(self.hier_widget, area='bottom')
        # self.viewer.window.add_dock_widget(self.TreeWidget, area="bottom",name="TreeWidget")

        tabwidget_bottom = QTabWidget()
        tabwidget_bottom.addTab(self.TreeWidget, "TreeWidget")
        tabwidget_bottom.addTab(self.hier_widget.native, "Hierarchy")
        self.viewer.window.add_dock_widget(tabwidget_bottom, area='bottom')

        #Connect to signals
        self.NavigationWidget.change_chunk.connect(self.update_chunk_from_button)
        self.NavigationWidget.goto_frame.connect(self.update_chunk_from_frame)
        self.EditingMenu.add_cell_button_pressed.connect(self.add_cell_from_database)

        #Connect to napari's time slider (dims) event)
        self.viewer.dims.events.current_step.connect(self.on_dims_changed)

        #Connect red flag UI buttons
        self.current_red_flag_index = 0
        self.NavigationWidget.red_flag_prev_btn.clicked.connect(self.go_to_prev_red_flag)
        self.NavigationWidget.red_flag_next_btn.clicked.connect(self.go_to_next_red_flag)
        self.NavigationWidget.red_flag_ignore_btn.clicked.connect(self.ignore_red_flag)
        self.NavigationWidget.red_flag_counter.clicked.connect(self.goto_red_flag)
        self.NavigationWidget.tracks_viewer.tracks_updated.connect(self.update_red_flags)

        self.add_tracks()
        self.NavigationWidget.update_chunk_label()
        self.update_red_flag_counter_and_info()

    def update_red_flags(self):
        self.databasehandler.recompute_red_flags()
        self.update_red_flag_counter_and_info()

    def update_red_flag_counter_and_info(self):
        """Update the red flag label to show the current red flag index and total count."""
        total = len(self.databasehandler.red_flags)
        if total > 0:
            # Display indices as 1-indexed (e.g., "3/80")
            self.NavigationWidget.red_flag_counter.setText(f"{self.current_red_flag_index + 1}/{total}")
            df_rf = self.databasehandler.red_flags.iloc[[self.current_red_flag_index]]
            text = f"{df_rf.iloc[0].id} {df_rf.iloc[0].event} at t={df_rf.iloc[0].t}"
            self.NavigationWidget.red_flag_info.setText(text)
        else:
            self.NavigationWidget.red_flag_counter.setText("0/0")
            self.NavigationWidget.red_flag_info.setText("-")

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
        self.link_layers()
        self.viewer.layers[self.databasehandler.name+'_seg'].iso_gradient_mode = "smooth"
        if "hierarchy" in self.viewer.layers:
            self.viewer.layers['hierarchy'].visible = False
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
        self.update_pop_add_hierarchy_layer()

        self.add_tracks()

        if direction == 'prev':
            desired_chunk_time = self.databasehandler.time_chunk_length - self.databasehandler.time_chunk_overlap + current_slider_position - 1
        else:
            desired_chunk_time = -self.databasehandler.time_chunk_length + self.databasehandler.time_chunk_overlap + current_slider_position + 1

        self.set_time_slider(desired_chunk_time)       
        self.NavigationWidget.update_chunk_label()


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
        self.update_pop_add_hierarchy_layer()

        if cur_chunk != new_chunk:
            self.add_tracks()        

        chunk_frame = frame - self.databasehandler.time_chunk_starts[self.databasehandler.time_chunk]
        self.set_time_slider(chunk_frame)
        self.NavigationWidget.update_chunk_label()
        self.update_time_label()

    def set_time_slider(self, chunk_frame):
        self.viewer.dims.current_step = (chunk_frame, *self.viewer.dims.current_step[1:])

    def on_dims_changed(self, _):
        self.update_time_label()

    def update_time_label(self):
        chunk = self.databasehandler.time_chunk
        cur_frame = self.viewer.dims.current_step[0]
        cur_world_time = cur_frame + self.databasehandler.time_chunk_starts[chunk]

        self.NavigationWidget.time_input.setText(str(cur_world_time))

    def check_navigation_button_validity(self):
        #enable/disable buttons if on first/last chunk
        chunk = self.databasehandler.time_chunk
        if chunk == 0:
            self.NavigationWidget.time_prev_btn.setEnabled(False)
        else:
            self.NavigationWidget.time_prev_btn.setEnabled(True)

        if chunk == self.databasehandler.num_time_chunks - 1:
            self.NavigationWidget.time_next_btn.setEnabled(False)
        else:
            self.NavigationWidget.time_next_btn.setEnabled(True)

    def link_layers(self):
        layer_names = [self.databasehandler.name + type for type in ['_seg','_tracks','_points']]
        layers_to_link = [layer for layer in self.viewer.layers if layer.name in layer_names]
        self.viewer.layers.link_layers(layers_to_link)  

    def update_pop_add_hierarchy_layer(self):
        self.hier_widget.ultrack_array.set_time_window(self.databasehandler.time_window)
        self.viewer.layers.pop('hierarchy')
        self.viewer.add_labels(self.hier_widget.ultrack_array, name='hierarchy', scale=self.databasehandler.scale)
        self.viewer.layers['hierarchy'].visible = False
        self.viewer.layers.move(3,0)
        self.viewer.layers.selection.active = self.viewer.layers[self.databasehandler.name+'_seg']   #select segmentation layer

    def add_cell_from_database(self, node_id: int):
        add_flag = False
        #check if node_id exists in database
        try:
            time = get_node_values(self.databasehandler.config_adjusted.data_config, node_id, NodeDB.t)
            add_flag = True
        except:
            show_warning("Cell does not exist in database")

        #check if node_is is already in solution (selected==1), but only check if node_id exists in database
        if add_flag:
            selected = get_node_values(self.databasehandler.config_adjusted.data_config, node_id, NodeDB.selected)
            if selected == 1:
                add_flag = False
                show_warning("Cell is already in solution")
                self.EditingMenu.cell_input.setText('')

        if add_flag:
            #move to the respective chunk of the added cell
            self.update_chunk_from_frame(time)

            max_track_id = max(self.NavigationWidget.tracks_viewer.tracks_controller.tracks.track_id_to_node.keys())
            time_in_chunk = time - self.databasehandler.time_window[0]
            pixels = [(np.array([0,0,0]))]  #provide mock pixels, to make sure tracks_controller doesn't make a new node_id...
            attributes = {
                    NodeAttr.TIME.value: [time_in_chunk],
                    NodeAttr.TRACK_ID.value: [max_track_id+1],
                    "node_id": [node_id],
            }

            self.NavigationWidget.tracks_viewer.tracks_controller.add_nodes(attributes,pixels)
            #ToDo: this is probably wrong, because graph.node attributes are set after _add_nodes is used, to graph nodes do not have time attribute (used to check if track_id already exists in TC._add_nodes)
            self.EditingMenu.cell_input.setText('')

def wrap_default_widgets_in_tabs(viewer):
    # -- 1) Identify the default dock widgets by going up the parent chain.
    # For controls: the dock widget is the direct parent.
    controls_dock = viewer.window.qt_viewer.controls.parentWidget()
    # For the layer list: go up two levels.
    list_dock = viewer.window.qt_viewer.layers.parentWidget().parentWidget()

    # -- 2) Instead of only taking the inner widget,
    # retrieve the entire container from the dock widget.
    controls_container = controls_dock.widget() if controls_dock else None
    list_container = list_dock.widget() if list_dock else None

    # -- 3) Remove the dock widgets from the main window,
    # but do not close or delete them so that Napari’s internal references remain valid.
    main_window = viewer.window._qt_window
    for dock in [controls_dock, list_dock]:
        if dock is not None:
            main_window.removeDockWidget(dock)

    # -- 4) Detach the container widgets from their docks
    if controls_container:
        controls_container.setParent(None)
    if list_container:
        list_container.setParent(None)

    # -- 5) Create a tab widget and add the containers as tabs.
    tab_widget = QTabWidget()
    if controls_container:
        tab_widget.addTab(controls_container, "Layer Controls")
    if list_container:
        tab_widget.addTab(list_container, "Layer List")

    # -- 6) Add our new tab widget as a dock widget.
    new_dock = viewer.window.add_dock_widget(tab_widget, area="left",name='napari')

    # -- 7) (Optional) Update internal viewer references so that
    # Napari’s menu actions refer to the new widgets.
    viewer.window.qt_viewer._controls = controls_container
    viewer.window.qt_viewer._layers = list_container

    # (Optional) Also update the internal dict of dock widgets if needed:
    # viewer.window._dock_widgets['Layer List'] = new_dock
