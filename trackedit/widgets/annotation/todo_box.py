import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QPushButton, QHBoxLayout, QVBoxLayout
from trackedit.widgets.ClickableLabel import ClickableLabel
from trackedit.widgets.base_box import NavigationBox

class TodoAnnotationBox(NavigationBox):

    update_chunk_from_frame_signal = Signal(int)

    def __init__(self, tracks_viewer, databasehandler):
        super().__init__("Annotations", max_height=200)
        self.tracks_viewer = tracks_viewer
        self.databasehandler = databasehandler
        self.current_annotation_index = 0

        # Create controls
        self.todoannotation_counter = ClickableLabel("0/0")
        self.todoannotation_counter.setFixedWidth(50)
        
        self.todoannotation_prev_btn = QPushButton("<")
        self.todoannotation_prev_btn.setFixedWidth(30)
        
        self.todoannotation_next_btn = QPushButton(">")
        self.todoannotation_next_btn.setFixedWidth(30)

        # Layout
        todoannotation_layout = QHBoxLayout()
        todoannotation_layout.addWidget(self.todoannotation_prev_btn)
        todoannotation_layout.addWidget(self.todoannotation_counter)
        todoannotation_layout.addWidget(self.todoannotation_next_btn)
        
        self.layout.addLayout(todoannotation_layout)
        self.layout.setAlignment(todoannotation_layout, Qt.AlignLeft)

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
        self.todoannotation_prev_btn.clicked.connect(self.go_to_prev_annotation)
        self.todoannotation_next_btn.clicked.connect(self.go_to_next_annotation)
        self.todoannotation_counter.clicked.connect(self.goto_annotation)
        self.hair_btn.clicked.connect(self.on_hair_clicked)
        self.support_btn.clicked.connect(self.on_support_clicked)
        self.mantle_btn.clicked.connect(self.on_mantle_clicked)
        self.tracks_viewer.tracks_updated.connect(self.update_todoannotation)
        # self.tracks_viewer.selected_nodes.list_updated.connect(self._check_selected_node_matches_annotation)

    def update_todoannotation(self):
        """Update the annotations and the annotation counter"""
        self.databasehandler.recompute_todoannotations()
        self.update_annotation_counter()
        pass

    def update_annotation_counter(self):
        """Update the annotation counter to show the current annotation index and total count."""
        total = len(self.databasehandler.todoannotations)
        if total > 0:
            self.todoannotation_counter.setText(f"{self.current_annotation_index + 1}/{total}")
        else:
            self.todoannotation_counter.setText("0/0")
        pass

    def go_to_next_annotation(self):
        """Navigate to the next annotation in the list and jump to that timepoint."""
        # total = len(self.databasehandler.todoannotations)
        # if total == 0:
        #     return
        # self.current_annotation_index = (self.current_annotation_index + 1) % total
        # self.goto_annotation()
        pass

    def go_to_prev_annotation(self):
        """Navigate to the previous annotation in the list and jump to that timepoint."""
        # total = len(self.databasehandler.todoannotations)
        # if total == 0:
        #     return
        # self.current_annotation_index = (self.current_annotation_index - 1) % total
        # self.goto_annotation()
        pass

    def goto_annotation(self):
        """Jump to the time of the current annotation."""
        # annotation_time = int(self.databasehandler.todoannotations.iloc[self.current_annotation_index]["t"])
        # self.update_chunk_from_frame_signal.emit(annotation_time)

        # #update the selected nodes in the TreeWidget
        # label = self.databasehandler.todoannotations.iloc[self.current_annotation_index]["id"]
        # self.tracks_viewer.selected_nodes._list = []
        # self.tracks_viewer.selected_nodes._list.append(label)
        # self.tracks_viewer.selected_nodes.list_updated.emit()

        # self.update_annotation_counter()
        pass

    def _check_selected_node_matches_annotation(self):
        # """Check if the selected node matches the current annotation label."""
        # selected_nodes = self.tracks_viewer.selected_nodes._list

        # if len(selected_nodes) != 1:
        #     self.todoannotation_counter.setStyleSheet("color: gray;")
        #     return

        # selected_node = selected_nodes[0]
        # annotation_ids = self.databasehandler.todoannotations['id'].values
        
        # try:
        #     index = np.where(annotation_ids == selected_node)[0][0]
        #     self.current_annotation_index = index
        #     self.todoannotation_counter.setText(f"{index + 1}/{len(self.databasehandler.todoannotations)}")
        #     self.todoannotation_counter.setStyleSheet("")
        # except IndexError:
        #     self.todoannotation_counter.setStyleSheet("color: gray;") 
        pass

    def on_hair_clicked(self):
        print("Hair button clicked")
        self.annotate_track("hair")
        
    def on_support_clicked(self):
        print("Support button clicked")
        self.annotate_track("support")
        
    def on_mantle_clicked(self):
        print("Mantle button clicked")
        self.annotate_track("mantle")

    def annotate_track(self, label: str):
        print(f"Annotating track with label: {label}")
        # self.databasehandler.annotate_track(label)