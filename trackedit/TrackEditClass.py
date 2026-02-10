import napari
import numpy as np
from motile_toolbox.candidate_graph import NodeAttr
from napari.utils.colormaps import DirectLabelColormap
from napari.utils.notifications import show_info, show_warning
from qtpy.QtWidgets import QTabWidget
from ultrack.core.database import NodeDB, get_node_values
from ultrack.core.interactive import add_new_node

from motile_tracker.data_model.solution_tracks import SolutionTracks
from motile_tracker.data_views import TracksViewer, TreeWidget
from trackedit.DatabaseHandler import DatabaseHandler
from trackedit.utils.utils import create_cell_mask_and_bbox, fix_overlap_ancestor_ids
from trackedit.widgets.annotation.annotation_widget import AnnotationWidget
from trackedit.widgets.CustomEditingWidget import CustomEditingMenu
from trackedit.widgets.HierarchyWidget import HierarchyVizWidget
from trackedit.widgets.NavigationWidget import NavigationWidget


class TrackEditClass:
    def __init__(
        self,
        viewer: napari.Viewer,
        databasehandler: DatabaseHandler,
        flag_show_hierarchy: bool = True,
        flag_allow_adding_spherical_cell: bool = False,
    ):
        self.viewer = viewer
        self.viewer.layers.clear()  # Remove all existing layers
        self.databasehandler = databasehandler
        self.flag_show_hierarchy = flag_show_hierarchy

        self.tracksviewer = TracksViewer.get_instance(self.viewer)

        self.TreeWidget = TreeWidget(self.viewer)
        self.NavigationWidget = NavigationWidget(self.viewer, self.databasehandler)
        self.AnnotationWidget = AnnotationWidget(self.viewer, self.databasehandler)
        self.EditingMenu = CustomEditingMenu(
            self.viewer,
            self.databasehandler,
            allow_adding_spherical_cell=flag_allow_adding_spherical_cell,
        )

        tabwidget_right = QTabWidget()
        tabwidget_right.addTab(self.NavigationWidget, "Navigation")
        tabwidget_right.addTab(self.EditingMenu, "Edit Tracks")
        tabwidget_right.addTab(self.AnnotationWidget, "Annotations")
        tabwidget_right.setMaximumWidth(330)
        self.viewer.window.add_dock_widget(
            tabwidget_right, area="right", name="TrackEdit Widgets"
        )

        if self.flag_show_hierarchy:
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

            self.hier_widget.labels_layer.signals.click_on_hierarchy_cell.connect(
                self.EditingMenu.click_on_hierarchy_cell
            )

        # Add annotation layer to viewer
        self.viewer.add_labels(
            self.databasehandler.annotArray,
            name="annotations",
            scale=self.databasehandler.scale,
        )
        self.viewer.layers["annotations"].colormap = DirectLabelColormap(
            color_dict=self.databasehandler.color_mapping
        )

        # Add Imaging layers to viewer
        if self.databasehandler.imaging_flag:
            # Define colormap sequence
            colormaps = ["green", "red", "blue", "magenta", "cyan", "yellow"]

            for i, layer_name in enumerate(
                self.databasehandler.imagingArray.layer_names
            ):
                channel_data = self.databasehandler.imagingArray.get_channel_data(i)
                colormap = colormaps[i % len(colormaps)]
                opacity = 1.0 if i == 0 else 0.5  # First layer full opacity, others 0.5

                layer = self.viewer.add_image(
                    channel_data,
                    name=f"im_{layer_name}",
                    colormap=colormap,
                    opacity=opacity,
                    scale=self.databasehandler.scale,
                    visible=False,
                    translate=self.databasehandler.image_translate,
                )
                layer.reset_contrast_limits()

        tabwidget_bottom = QTabWidget()
        tabwidget_bottom.addTab(self.TreeWidget, "TreeWidget")
        if self.flag_show_hierarchy:
            tabwidget_bottom.addTab(self.hier_widget.native, "Hierarchy")
        self.viewer.window.add_dock_widget(tabwidget_bottom, area="bottom")

        # Connect to signals
        self.NavigationWidget.change_chunk.connect(self.update_chunk_from_button)
        self.NavigationWidget.goto_frame.connect(self.update_chunk_from_frame)
        self.NavigationWidget.tmax_did_change.connect(self.update_chunk_from_frame)
        self.NavigationWidget.division_box.update_chunk_from_frame_signal.connect(
            self.update_chunk_from_frame
        )
        self.NavigationWidget.red_flag_box.update_chunk_from_frame_signal.connect(
            self.update_chunk_from_frame
        )
        self.AnnotationWidget.toannotate_box.update_chunk_from_frame_signal.connect(
            self.update_chunk_from_frame
        )
        self.EditingMenu.add_cell_button_pressed.connect(self.add_cell_from_database)
        self.EditingMenu.duplicate_cell_button_pressed.connect(
            self.duplicate_cell_from_database
        )

        # Connect spherical cell signal if feature is enabled
        if flag_allow_adding_spherical_cell:
            self.EditingMenu.add_spherical_cell_toggled.connect(
                self._toggle_add_cell_mode
            )
            # Initialize spherical cell mode flag
            self._add_cell_mode_active = False

        self.add_tracks()
        self.NavigationWidget.time_box.update_chunk_label()
        self.NavigationWidget.red_flag_box.update_red_flag_counter_and_info()
        self.NavigationWidget.division_box.update_division_counter()

    # ===============================================
    # Add tracks
    # ===============================================

    def add_tracks(self):
        """Add a solution set of tracks to the tracks viewer results list"""

        # Set dimensions based on actual data shape
        if self.databasehandler.ndim == 4:  # TZYX
            pos_attr = ("z", "y", "x")
        elif self.databasehandler.ndim == 3:  # TYX
            pos_attr = ("y", "x")
        else:
            raise ValueError(
                f"Expected dataset with 3 or 4 dimensions, T(Z)YX. Found {self.databasehandler.ndim}."
            )

        tracks = SolutionTracks(
            graph=self.databasehandler.nxgraph,
            segmentation=self.databasehandler.segments,
            pos_attr=pos_attr,
            time_attr="t",
            scale=[1, *self.databasehandler.scale],
        )

        # Add tracks and verify layer creation
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
        self.update_hierarchy_layer()
        if self.databasehandler.imaging_flag:
            for i, layer_name in enumerate(
                self.databasehandler.imagingArray.layer_names
            ):
                channel_data = self.databasehandler.imagingArray.get_channel_data(i)
                self.viewer.layers[f"im_{layer_name}"].data = channel_data

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

        # Force colormap update
        current_colormap = self.viewer.layers["ultrack_seg"].colormap
        self.viewer.layers["ultrack_seg"].colormap = DirectLabelColormap(
            color_dict=current_colormap.color_dict
        )

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
        self.update_hierarchy_layer()
        if self.databasehandler.imaging_flag:
            for i, layer_name in enumerate(
                self.databasehandler.imagingArray.layer_names
            ):
                channel_data = self.databasehandler.imagingArray.get_channel_data(i)
                self.viewer.layers[f"im_{layer_name}"].data = channel_data

        # Update tracks if chunks are different OR if this was triggered by a Tmax change
        # TODO: not sure this is the best way to do this
        if cur_chunk != new_chunk or frame == self.databasehandler.Tmax - 1:
            self.add_tracks()

        chunk_frame = (
            frame
            - self.databasehandler.time_chunk_starts[self.databasehandler.time_chunk]
        )
        self.NavigationWidget.time_box.set_time_slider(chunk_frame)
        self.NavigationWidget.time_box.update_chunk_label()
        self.NavigationWidget.time_box.update_time_label()

        # Force colormap update
        current_colormap = self.viewer.layers["ultrack_seg"].colormap
        self.viewer.layers["ultrack_seg"].colormap = DirectLabelColormap(
            color_dict=current_colormap.color_dict
        )

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

    def update_hierarchy_layer(self):
        """Update the hierarchy layer with the new chunk."""
        if self.flag_show_hierarchy:
            self.hier_widget.ultrack_array.set_time_window(
                self.databasehandler.time_window
            )
            self.hier_widget.labels_layer.refresh()  # This should trigger proper update while maintaining callbacks
            self.viewer.layers.selection.active = self.viewer.layers[
                self.databasehandler.name + "_seg"
            ]
        else:
            pass

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
        except Exception:
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
            # ToDo: this is probably wrong, because graph.node attributes are set after _add_nodes is used,
            # so graph nodes do not have (correct) time attribute
            # (used to check if track_id already exists in TC._add_nodes)
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
        except Exception:
            show_warning("Cell does not exist in database")

        # check if node_is is already in solution (selected==1), but only check if node_id exists in database
        if add_flag:
            time_original = get_node_values(
                self.databasehandler.config_adjusted.data_config, node_id, NodeDB.t
            )
            if time_original == time:
                add_flag = False
                self.EditingMenu.duplicate_cell_id_input.setText("")
                self.EditingMenu.duplicate_time_input.setText("")
                self.EditingMenu.add_cell_input.setText("")
                if time_original == time:
                    show_warning(
                        "Cell is from this time point, use 'Add Cell' field above to add this cell,"
                        " it is not a duplication"
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
                include_overlaps=True,
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

    def add_spherical_cell_at_position(self, position_scaled, radius_pixels=10):
        """Add a new cell with spherical segmentation at the given position.

        Parameters
        ----------
        position_scaled : array-like
            Position in viewer coordinates (scaled)
        radius_pixels : float
            Radius of the sphere in pixels (default: 10)

        Returns
        -------
        new_node_id : int or None
            Database ID of the newly created node, or None if failed
        """
        current_time = self.viewer.dims.current_step[0]
        self.update_chunk_from_frame(current_time)

        # Create mask and bbox
        mask, bbox = create_cell_mask_and_bbox(
            position_scaled=position_scaled,
            radius_pixels=radius_pixels,
            ndim=self.databasehandler.ndim,
            scale=(
                self.databasehandler.z_scale,
                self.databasehandler.y_scale,
                self.databasehandler.x_scale,
            )
            if self.databasehandler.ndim == 4
            else (self.databasehandler.y_scale, self.databasehandler.x_scale),
            data_shape_full=self.databasehandler.data_shape_full,
        )
        if mask is None:
            return None

        # Add to database
        try:
            new_node_id = add_new_node(
                self.databasehandler.config_adjusted,
                time=current_time,
                mask=mask,
                bbox=bbox,
                include_overlaps=True,
            )
            fix_overlap_ancestor_ids(
                database_path=self.databasehandler.config_adjusted.data_config.database_path,
                new_node_id=new_node_id,
                current_time=current_time,
            )
        except Exception as e:
            show_warning(f"Failed to add node to database: {e}")
            import traceback

            traceback.print_exc()
            return None

        # Add to tracking system
        track_ids = (
            self.NavigationWidget.tracks_viewer.tracks_controller.tracks.track_id_to_node.keys()
        )
        max_track_id = max(track_ids) if track_ids else 0
        time_in_chunk = current_time - self.databasehandler.time_window[0]

        attributes = {
            NodeAttr.TIME.value: [time_in_chunk],
            NodeAttr.TRACK_ID.value: [max_track_id + 1],
            "node_id": [new_node_id],
        }
        self.tracksviewer.tracks_controller.add_nodes(
            attributes, [(np.array([0, 0, 0]))]
        )

        # Refresh and auto-disable
        show_info(f"Added spherical cell with ID {new_node_id}")
        self.databasehandler.segments.force_refill()
        self.viewer.layers[self.databasehandler.name + "_seg"].refresh()

        # Auto-disable only if button exists
        if hasattr(self.EditingMenu, "add_spherical_cell_btn"):
            from qtpy.QtCore import QTimer

            QTimer.singleShot(
                0,
                lambda: (
                    self.EditingMenu.add_spherical_cell_btn.setChecked(False),
                    self._toggle_add_cell_mode(False),
                ),
            )

        return new_node_id

    def _toggle_add_cell_mode(self, checked):
        """Toggle the add spherical cell mode on/off."""
        self._add_cell_mode_active = checked

        if checked:
            # Only add callback if not already present
            if self._on_mouse_click_add_cell not in self.viewer.mouse_drag_callbacks:
                self.viewer.mouse_drag_callbacks.append(self._on_mouse_click_add_cell)
        else:
            # Remove ALL instances of the callback (in case of duplicates)
            while self._on_mouse_click_add_cell in self.viewer.mouse_drag_callbacks:
                try:
                    self.viewer.mouse_drag_callbacks.remove(
                        self._on_mouse_click_add_cell
                    )
                except ValueError:
                    break

    def _on_mouse_click_add_cell(self, viewer, event):
        """Handle mouse click when add cell mode is active.

        Called when user clicks in viewer while add cell mode is on.
        """
        # Guard: only proceed if mode is actually active
        if not self._add_cell_mode_active:
            return

        # Only trigger on click (not drag)
        if event.type == "mouse_press":
            # Get click position in data coordinates
            # Position includes time dimension: (t, z, y, x) or (t, y, x)
            position = viewer.cursor.position

            # Extract spatial coordinates (remove time)
            if self.databasehandler.ndim == 4:
                # 4D data: (t, z, y, x) -> (z, y, x)
                position_spatial = position[1:]
            else:
                # 3D data: (t, y, x) -> (y, x)
                position_spatial = position[1:]

            # Add spherical cell at clicked position
            self.add_spherical_cell_at_position(position_spatial)

            # Yield to prevent further event propagation
            yield
