import napari
import numpy as np
from motile_toolbox.candidate_graph import NodeAttr
from napari.utils.notifications import show_warning
from qtpy.QtWidgets import QTabWidget
from ultrack.core.database import NodeDB, get_node_values
from ultrack.core.interactive import add_new_node

from motile_tracker.data_model.solution_tracks import SolutionTracks
from motile_tracker.data_views import TracksViewer, TreeWidget
from trackedit.DatabaseHandler import DatabaseHandler
from trackedit.widgets.CustomEditingMenu import CustomEditingMenu
from trackedit.widgets.HierarchyWidget import HierarchyVizWidget
from trackedit.widgets.NavigationWidget import NavigationWidget


class TrackEditClass:
    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        self.viewer = viewer
        self.databasehandler = databasehandler
        self.tracksviewer = TracksViewer.get_instance(self.viewer)

        self.TreeWidget = TreeWidget(self.viewer)
        self.NavigationWidget = NavigationWidget(self.viewer, self.databasehandler)
        self.EditingMenu = CustomEditingMenu(self.viewer, self.databasehandler)

        tabwidget_right = QTabWidget()
        tabwidget_right.addTab(self.NavigationWidget, "Navigation")
        tabwidget_right.addTab(self.EditingMenu, "Edit Tracks")
        tabwidget_right.setMaximumWidth(330)
        self.viewer.window.add_dock_widget(
            tabwidget_right, area="right", name="TrackEdit Widgets"
        )

        self.hier_widget = HierarchyVizWidget(
            viewer=viewer,
            scale=self.databasehandler.scale,
            config=self.databasehandler.config_adjusted,
        )
        hier_shape = self.hier_widget.ultrack_array.shape
        tmax = self.databasehandler.data_shape_chunk[0]
        self.hier_widget.ultrack_array.shape = (tmax, *hier_shape[1:])

        # Store reference to the existing hierarchy layer
        self.hierarchy_layer = self.hier_widget.labels_layer
        self.hierarchy_layer.visible = False

        tabwidget_bottom = QTabWidget()
        tabwidget_bottom.addTab(self.TreeWidget, "TreeWidget")
        tabwidget_bottom.addTab(self.hier_widget.native, "Hierarchy")
        self.viewer.window.add_dock_widget(tabwidget_bottom, area="bottom")

        # Connect to signals
        self.NavigationWidget.change_chunk.connect(self.update_chunk_from_button)
        self.NavigationWidget.goto_frame.connect(self.update_chunk_from_frame)
        self.NavigationWidget.division_box.update_chunk_from_frame_signal.connect(
            self.update_chunk_from_frame
        )
        self.NavigationWidget.red_flag_box.update_chunk_from_frame_signal.connect(
            self.update_chunk_from_frame
        )
        self.EditingMenu.add_cell_button_pressed.connect(self.add_cell_from_database)
        self.EditingMenu.duplicate_cell_button_pressed.connect(
            self.duplicate_cell_from_database
        )
        self.hier_widget.labels_layer.signals.click_on_hierarchy_cell.connect(
            self.EditingMenu.click_on_hierarchy_cell
        )

        self.add_tracks()
        self.NavigationWidget.time_box.update_chunk_label()
        self.NavigationWidget.red_flag_box.update_red_flag_counter_and_info()
        self.NavigationWidget.division_box.update_division_counter()

    # ===============================================
    # Add tracks
    # ===============================================

    def add_tracks(self):
        """Add a solution set of tracks to the tracks viewer results list

        Args:
            tracker (ultrack.Tracker): the ultrack tracker containing the solution
            name (str): the display name of the solution tracks
        """

        # create tracks object
        tracks = SolutionTracks(
            graph=self.databasehandler.nxgraph,
            segmentation=self.databasehandler.segments,
            pos_attr=("z", "y", "x"),
            time_attr="t",
            scale=[
                1,
                *self.databasehandler.scale,
            ],  # db.scale is zyx, SolutionTracks needs tzyx
        )

        # add tracks to viewer
        self.tracksviewer.tracks_list.add_tracks(tracks, name=self.databasehandler.name)
        self.viewer.layers.selection.active = self.viewer.layers[
            self.databasehandler.name + "_seg"
        ]  # select segmentation layer

        self.NavigationWidget.time_box.check_navigation_button_validity()
        self.link_layers()
        self.viewer.layers[
            self.databasehandler.name + "_seg"
        ].iso_gradient_mode = "smooth"
        if "hierarchy" in self.viewer.layers:
            self.viewer.layers["hierarchy"].visible = False
        # ToDo: check if all tracks are added or overwritten

    # ===============================================
    # Navigation
    # ===============================================

    def update_chunk_from_button(self, direction: str):
        cur_chunk = self.databasehandler.time_chunk
        current_slider_position = self.viewer.dims.current_step[0]

        if direction not in ["prev", "next"]:
            raise ValueError(f"Invalid direction: {direction}")

        # change the time chunk index
        if direction == "prev":
            new_chunk = cur_chunk - 1
        else:
            new_chunk = cur_chunk + 1

        # check if the new chunk is within the limits
        if new_chunk < 0:
            new_chunk = 0
        elif new_chunk == self.databasehandler.num_time_chunks:
            new_chunk = self.databasehandler.num_time_chunks - 1

        self.databasehandler.set_time_chunk(new_chunk)
        self.update_pop_add_hierarchy_layer()

        self.add_tracks()

        if direction == "prev":
            desired_chunk_time = (
                self.databasehandler.time_chunk_length
                - self.databasehandler.time_chunk_overlap
                + current_slider_position
                - 1
            )
        else:
            desired_chunk_time = (
                -self.databasehandler.time_chunk_length
                + self.databasehandler.time_chunk_overlap
                + current_slider_position
                + 1
            )

        self.NavigationWidget.time_box.set_time_slider(desired_chunk_time)
        self.NavigationWidget.time_box.update_chunk_label()

    def update_chunk_from_frame(self, frame: int):
        """Handle navigation by a user-entered time frame.

        This calculates the chunk containing the given frame.
        For example, if each chunk is 100 frames, frame 235 belongs in chunk 2.
        """
        if frame < 0:
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

        chunk_frame = (
            frame
            - self.databasehandler.time_chunk_starts[self.databasehandler.time_chunk]
        )
        self.NavigationWidget.time_box.set_time_slider(chunk_frame)
        self.NavigationWidget.time_box.update_chunk_label()
        self.NavigationWidget.time_box.update_time_label()

    # ===============================================
    # Linking/layer stuff
    # ===============================================

    def link_layers(self):
        """Link the segmentation, tracks, and points layers of the Motile widget together."""
        layer_names = [
            self.databasehandler.name + type for type in ["_seg", "_tracks", "_points"]
        ]
        layers_to_link = [
            layer for layer in self.viewer.layers if layer.name in layer_names
        ]
        self.viewer.layers.link_layers(layers_to_link)

    def update_pop_add_hierarchy_layer(self):
        """Update the hierarchy layer with the new chunk."""
        self.hier_widget.ultrack_array.set_time_window(self.databasehandler.time_window)
        self.hier_widget.labels_layer.refresh()  # This should trigger proper update while maintaining callbacks
        self.viewer.layers.selection.active = self.viewer.layers[
            self.databasehandler.name + "_seg"
        ]

    # ===============================================
    # Add cells
    # ===============================================

    def add_cell_from_database(self, node_id: int):
        add_flag = False
        # check if node_id exists in database
        try:
            time = get_node_values(
                self.databasehandler.config_adjusted.data_config, node_id, NodeDB.t
            )
            add_flag = True
        except:
            show_warning("Cell does not exist in database")

        # check if node_is is already in solution (selected==1), but only check if node_id exists in database
        if add_flag:
            selected = get_node_values(
                self.databasehandler.config_adjusted.data_config,
                node_id,
                NodeDB.selected,
            )
            if selected == 1:
                add_flag = False
                show_warning("Cell is already in solution")
                self.EditingMenu.add_cell_input.setText("")
                self.EditingMenu.duplicate_cell_id_input.setText("")
                self.EditingMenu.duplicate_time_input.setText("")

        if add_flag:
            # move to the respective chunk of the added cell
            self.update_chunk_from_frame(time)

            max_track_id = max(
                self.tracksviewer.tracks_controller.tracks.track_id_to_node.keys()
            )
            time_in_chunk = time - self.databasehandler.time_window[0]
            pixels = [
                (np.array([0, 0, 0]))
            ]  # provide mock pixels, to make sure tracks_controller doesn't make a new node_id...
            attributes = {
                NodeAttr.TIME.value: [time_in_chunk],
                NodeAttr.TRACK_ID.value: [max_track_id + 1],
                "node_id": [node_id],
            }

            self.tracksviewer.tracks_controller.add_nodes(attributes, pixels)
            # ToDo: this is probably wrong, because graph.node attributes are set after _add_nodes is used, so graph nodes do not have (correct) time attribute (used to check if track_id already exists in TC._add_nodes)
            self.EditingMenu.add_cell_input.setText("")
            self.EditingMenu.duplicate_cell_id_input.setText("")
            self.EditingMenu.duplicate_time_input.setText("")

    def duplicate_cell_from_database(self, node_id: int, time: int):
        # ToDo: merge with previous function
        add_flag = False
        # check if node_id exists in database
        try:
            _ = get_node_values(
                self.databasehandler.config_adjusted.data_config, node_id, NodeDB.z
            )
            add_flag = True
        except:
            show_warning("Cell does not exist in database")

        # check if node_is is already in solution (selected==1), but only check if node_id exists in database
        if add_flag:
            selected = get_node_values(
                self.databasehandler.config_adjusted.data_config,
                node_id,
                NodeDB.selected,
            )
            time_original = get_node_values(
                self.databasehandler.config_adjusted.data_config, node_id, NodeDB.t
            )
            if (selected == True) or (time_original == time):
                add_flag = False
                self.EditingMenu.duplicate_cell_id_input.setText("")
                self.EditingMenu.duplicate_time_input.setText("")
                self.EditingMenu.add_cell_input.setText("")
                if selected == True:
                    show_warning(f"Cell is already in solution at this time {time}")
                if time_original == time:
                    show_warning(
                        f"Cell is from this time point, use 'Add Cell' field above to add this cell, it is not a duplication"
                    )

        if add_flag:
            # move to the respective chunk of the added cell
            self.update_chunk_from_frame(time)
            pickle = get_node_values(
                self.databasehandler.config_adjusted.data_config,
                [int(node_id)],
                NodeDB.pickle,
            )
            new_id = add_new_node(
                self.databasehandler.config_adjusted,
                time=time,
                mask=pickle.mask,
                bbox=pickle.bbox,
                include_overlaps=False,
            )
            print("added node to db:", new_id, "at time", time)

            max_track_id = max(
                self.NavigationWidget.tracks_viewer.tracks_controller.tracks.track_id_to_node.keys()
            )
            time_in_chunk = time - self.databasehandler.time_window[0]
            pixels = [(np.array([0, 0, 0]))]
            attributes = {
                NodeAttr.TIME.value: [time_in_chunk],
                NodeAttr.TRACK_ID.value: [max_track_id + 1],
                "node_id": [new_id],
            }

            self.tracksviewer.tracks_controller.add_nodes(attributes, pixels)
            self.EditingMenu.duplicate_cell_id_input.setText("")
            self.EditingMenu.duplicate_time_input.setText("")
