from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from typing import TypeAlias
else:
    # For Python < 3.10
    TypeAlias = str  # or = Any

import numpy as np
import pyqtgraph as pg
from motile_toolbox.candidate_graph import EdgeAttr, NodeAttr
from napari.utils.notifications import show_warning
from qtpy.QtGui import QColor
from ultrack.core.database import NodeDB, get_node_values

from motile_tracker.data_model.actions import ActionGroup, AddEdges, DeleteEdges
from motile_tracker.data_model.solution_tracks import SolutionTracks
from motile_tracker.data_model.tracks_controller import TracksController
from motile_tracker.data_views import TracksViewer
from motile_tracker.data_views.views.tree_view.tree_widget import TreePlot

AttrValue: TypeAlias = Any
AttrValues: TypeAlias = Sequence[AttrValue]
Attrs: TypeAlias = Mapping[str, AttrValues]
Node: TypeAlias = int
Edge: TypeAlias = tuple[Node, Node]


def create_db_add_nodes(DB_handler):
    def db_add_nodes(self):
        # don't use full old function, because it includes painting pixels in segmentation
        print("AddNodes:", self.nodes)
        if len(self.nodes) == 0:  # Skip if no nodes to add
            return

        # overwrite self.positions with values from database, scaled with z_scale
        new_pos = []
        for n in self.nodes:
            pos = get_node_values(
                DB_handler.config_adjusted.data_config,
                int(n),
                [NodeDB.z, NodeDB.y, NodeDB.x],
            )
            pos = pos.tolist()
            pos[0] *= DB_handler.z_scale
            new_pos.append(pos)
        self.positions = np.array(new_pos)

        self.tracks.add_nodes(
            self.nodes, self.times, self.positions, attrs=self.attributes
        )

        # Change the selected/annotation status of the nodes
        DB_handler.change_values(indices=self.nodes, field=NodeDB.selected, values=1)
        DB_handler.change_values(
            indices=self.nodes, field=NodeDB.generic, values=NodeDB.generic.default.arg
        )

    return db_add_nodes


def create_db_delete_nodes(DB_handler):
    def db_delete_nodes(self):
        print("DeleteNodes:", self.nodes)
        # don't use full old function, because it includes painting pixels in segmentation
        if len(self.nodes) == 0:  # Skip if no nodes to delete
            return

        DB_handler.clear_nodes_annotations(self.nodes)
        self.tracks.remove_nodes(self.nodes)

        # First disconnect orphaned children BEFORE we delete the nodes
        orphaned_children = DB_handler.df_full[
            DB_handler.df_full["parent_id"].isin(self.nodes)
        ].index.tolist()
        print("orphaned_children", orphaned_children)
        if orphaned_children:
            DB_handler.change_values(
                indices=orphaned_children, field=NodeDB.parent_id, values=-1
            )
            show_warning(
                "An edge in the next time window is removed, so 'UNDO' will not work."
            )
            # ToDo: potentially only remove orphan edges into the next time window,
            # because normal edges are already properly removed

        # Set nodes as unselected
        DB_handler.change_values(indices=self.nodes, field=NodeDB.selected, values=0)

    return db_delete_nodes


_old_add_edges_apply = AddEdges._apply


def create_db_add_edges(DB_handler):
    def db_add_edges(self):
        print("AddEdges:", self.edges)
        _old_add_edges_apply(self)

        if len(self.edges) == 0:  # Check length instead of direct boolean
            return

        DB_handler.clear_edges_annotations(self.edges)

        # Extract child nodes and parent nodes from edges
        child_nodes = [e[1] for e in self.edges]
        parent_nodes = [e[0] for e in self.edges]

        # Batch the changes into a single call
        DB_handler.change_values(
            indices=child_nodes, field=NodeDB.parent_id, values=parent_nodes
        )

    return db_add_edges


_old_delete_edges_apply = DeleteEdges._apply


def create_db_delete_edges(DB_handler):
    def db_delete_edges(self):
        print("DeleteEdges:", self.edges)
        _old_delete_edges_apply(self)

        if len(self.edges) == 0:  # Check length instead of direct boolean
            return

        # Extract child nodes from edges
        child_nodes = [e[1] for e in self.edges]

        # Batch the changes into a single call
        DB_handler.change_values(indices=child_nodes, field=NodeDB.parent_id, values=-1)

    return db_delete_edges


def _empty_compute_node_attrs(
    self, nodes: Iterable[Node], times: Iterable[int]
) -> Attrs:
    attrs: dict[str, list[Any]] = {
        NodeAttr.POS.value: [],
        NodeAttr.AREA.value: [],
    }
    for _ in nodes:
        attrs[NodeAttr.POS.value].append([0, 0, 0])
        attrs[NodeAttr.AREA.value].append(0)
    attrs[NodeAttr.POS.value] = np.array(attrs[NodeAttr.POS.value])
    return attrs


SolutionTracks._compute_node_attrs = _empty_compute_node_attrs


def _empty_compute_edge_attrs(self, edges: Iterable[Edge]) -> Attrs:
    attrs: dict[str, list[Any]] = {EdgeAttr.IOU.value: []}
    for _ in edges:
        attrs[EdgeAttr.IOU.value].append(0)
    return attrs


SolutionTracks._compute_edge_attrs = _empty_compute_edge_attrs


def empty_get_pixels(self, nodes: list[Node]) -> list[tuple[np.ndarray, ...]] | None:
    return []


SolutionTracks.get_pixels = empty_get_pixels


_old_tracks_viewer_refresh = TracksViewer._refresh


def create_tracks_viewer_and_segments_refresh(layer_name):
    def tracks_viewer_refresh_with_segments_refresh(
        self, node: str | None = None, refresh_view: bool = False
    ) -> None:
        _old_tracks_viewer_refresh(self, node, refresh_view)
        # refill and refresh the segments and annotations layers
        self.viewer.layers[layer_name + "_seg"].data.force_refill()
        self.viewer.layers[layer_name + "_seg"].refresh()
        self.viewer.layers["annotations"].data.force_refill()
        self.viewer.layers["annotations"].refresh()
        print("refreshed \n")

    return tracks_viewer_refresh_with_segments_refresh


# remove new edge that is created by delete_nodes to cover the gap
def my_delete_nodes(self, nodes: Iterable[None]):
    action_group1 = self._delete_nodes(nodes)

    # delete the edge that motile added over the gap after node deletion
    actions_merged = action_group1.actions
    for action in action_group1.actions:
        if isinstance(action, AddEdges):
            edges_to_delete = action.edges
            action_group2 = self._delete_edges(np.array(edges_to_delete))
            actions_merged = actions_merged + action_group2.actions
    action_group_together = ActionGroup(self.tracks, actions_merged)

    self.action_history.add_new_action(action_group_together)
    self.tracks.refresh.emit()


TracksController.delete_nodes = my_delete_nodes

# prevent skip edges (which are allowed by motile)
_old_is_valid = TracksController.is_valid


def is_valid_continuous(self, edge):
    # first check if the edge is continuous over time
    time0 = self.tracks.get_time(edge[0])
    time1 = self.tracks.get_time(edge[1])
    if (time1 - time0) > 1:
        show_warning(
            "Edge is rejected because it is not continuous over time (no skip edges allowed)."
        )
        return False, None

    # then check if the edge is valid in the usual way
    is_valid, valid_action = _old_is_valid(self, edge)

    return is_valid, valid_action


TracksController.is_valid = is_valid_continuous

_old_create_pyqtgraph_content = TreePlot._create_pyqtgraph_content


def patched_create_pyqtgraph_content(self, track_df, feature):
    """Patched version of _create_pyqtgraph_content to modify outline_pen."""
    # Call the original method
    _old_create_pyqtgraph_content(self, track_df, feature)

    # Overwrite the last line with transparency (alpha = 0)
    self.outline_pen = np.array(
        [pg.mkPen(QColor(150, 150, 150, 0)) for _ in range(len(self._pos))]
    )


TreePlot._create_pyqtgraph_content = patched_create_pyqtgraph_content

# def get_status(self, position, view_direction=None, dims_displayed=None, world=True):
#     return "True" #works to allow napari grid view, but not for cursor position/value display
# TrackLabels.get_status = get_status
