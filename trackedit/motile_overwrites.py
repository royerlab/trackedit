from typing import Any, Iterable, TypeAlias, Sequence, Mapping
from motile_toolbox.candidate_graph import NodeAttr, EdgeAttr
from motile_tracker.data_model.solution_tracks import SolutionTracks
from motile_tracker.data_views import TracksViewer   
from motile_tracker.data_views.views.tree_view.tree_widget import TreePlot
from motile_tracker.data_model.tracks_controller import TracksController
from motile_tracker.data_model.actions import AddEdges, DeleteEdges, ActionGroup
from ultrack.core.database import NodeDB, get_node_values, set_node_values

import numpy as np
import pyqtgraph as pg
from qtpy.QtGui import QColor

AttrValue: TypeAlias = Any
AttrValues: TypeAlias = Sequence[AttrValue]
Attrs: TypeAlias = Mapping[str, AttrValues]
Node: TypeAlias = int
Edge: TypeAlias = tuple[Node, Node]

def create_db_add_nodes(DB_handler):
    def db_add_nodes(self):
        # don't use full old function, because it includes painting pixels in segmentation

        #overwrite self.positions with values from database, scaled with z_scale
        new_pos = []
        for n in self.nodes:
            pos = get_node_values(DB_handler.config_adjusted.data_config, n, [NodeDB.z, NodeDB.y, NodeDB.x])
            pos = pos.tolist()
            pos[0] *= DB_handler.z_scale
            new_pos.append(pos)
        self.positions = np.array(new_pos)

        self.tracks.add_nodes(self.nodes, self.times, self.positions, attrs=self.attributes)
        print('AddNodes:',self.nodes)
        for n in self.nodes:
            DB_handler.change_value(index = n,
                        field = NodeDB.selected,
                        value = 1)
    return db_add_nodes
        
def create_db_delete_nodes(DB_handler):
    def db_delete_nodes(self):
        # don't use full old function, because it includes painting pixels in segmentation
        self.tracks.remove_nodes(self.nodes)
        print('DeleteNodes:',self.nodes)
        for n in self.nodes:
            DB_handler.change_value(index = n,
                        field = NodeDB.selected,
                        value = 0)
    return db_delete_nodes
    
_old_add_edges_apply = AddEdges._apply
def create_db_add_edges(DB_handler):
    def db_add_edges(self):
        _old_add_edges_apply(self)
        print('AddEdges:',self.edges)
        for e in self.edges:
            DB_handler.change_value(index = e[1],
                        field = NodeDB.parent_id,
                        value = e[0])
    return db_add_edges
            
_old_delete_edges_apply = DeleteEdges._apply
def create_db_delete_edges(DB_handler):
    def db_delete_edges(self):
        _old_delete_edges_apply(self)
        print('DeleteEdges:',self.edges)
        for e in self.edges:
            DB_handler.change_value(index = e[1],
                        field = NodeDB.parent_id,
                        value = -1)
    return db_delete_edges


def _empty_compute_node_attrs(self, nodes: Iterable[Node], times: Iterable[int]) -> Attrs:
    attrs: dict[str, list[Any]] = {
        NodeAttr.POS.value: [],
        NodeAttr.AREA.value: [],
    }
    for n in nodes:
        attrs[NodeAttr.POS.value].append([0,0,0])
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
    def tracks_viewer_refresh_with_segments_refresh(self, node: str | None = None, refresh_view: bool = False) -> None:
        _old_tracks_viewer_refresh(self, node, refresh_view)
        self.viewer.layers[layer_name + '_seg'].refresh()
        print('refreshed \n')
    return tracks_viewer_refresh_with_segments_refresh

#remove new edge that is created by delete_nodes to cover the gap
def my_delete_nodes(self,nodes: Iterable[None]):
    action_group1 = self._delete_nodes(nodes)

    #delete the edge that motile added over the gap after node deletion
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

_old_create_pyqtgraph_content = TreePlot._create_pyqtgraph_content
def patched_create_pyqtgraph_content(self, track_df, feature):
    """Patched version of _create_pyqtgraph_content to modify outline_pen."""
    # Call the original method
    _old_create_pyqtgraph_content(self, track_df, feature)

    # Overwrite the last line with transparency (alpha = 0)
    self.outline_pen = np.array(
        [pg.mkPen(QColor(150, 150, 150, 0)) for _ in range(len(self._pos))]
    )

# Monkey-patch the method
TreePlot._create_pyqtgraph_content = patched_create_pyqtgraph_content