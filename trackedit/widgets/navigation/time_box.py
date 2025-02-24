from napari.utils.notifications import show_warning
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton

from trackedit.widgets.navigation.base_box import NavigationBox


class TimeBox(NavigationBox):
    change_chunk = Signal(str)
    goto_frame = Signal(int)

    def __init__(self, viewer, databasehandler):
        super().__init__("Time navigation", max_height=120)
        self.viewer = viewer
        self.databasehandler = databasehandler

        # Define the buttons
        self.time_prev_btn = QPushButton("prev (<)")
        self.time_next_btn = QPushButton("next (>)")
        self.time_prev_btn.clicked.connect(self.press_prev)
        self.time_next_btn.clicked.connect(self.press_next)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.time_prev_btn)
        button_layout.addWidget(self.time_next_btn)

        # Define time input field
        self.time_input = QLineEdit()
        self.time_input.setPlaceholderText("Enter time")
        self.time_input.returnPressed.connect(self.on_time_input_entered)

        self.chunk_label = QLabel("temp. label")

        time_input_layout = QHBoxLayout()
        time_input_layout.addWidget(QLabel("time = "))
        time_input_layout.addWidget(self.time_input)
        time_input_layout.addWidget(self.chunk_label)

        self.layout.addLayout(time_input_layout)
        self.layout.addLayout(button_layout)

        # Connect to napari's time slider
        self.viewer.dims.events.current_step.connect(self.on_dims_changed)

    def set_time_slider(self, chunk_frame):
        self.viewer.dims.current_step = (
            chunk_frame,
            *self.viewer.dims.current_step[1:],
        )

    def on_dims_changed(self, _) -> None:
        self.update_time_label()

    def update_time_label(self) -> None:
        chunk = self.databasehandler.time_chunk
        cur_frame = self.viewer.dims.current_step[0]
        cur_world_time = cur_frame + self.databasehandler.time_chunk_starts[chunk]
        self.time_input.setText(str(cur_world_time))

    def check_navigation_button_validity(self) -> None:
        chunk = self.databasehandler.time_chunk
        self.time_prev_btn.setEnabled(chunk != 0)
        self.time_next_btn.setEnabled(chunk != self.databasehandler.num_time_chunks - 1)

    def press_prev(self):
        self.change_chunk.emit("prev")

    def press_next(self):
        self.change_chunk.emit("next")

    def on_time_input_entered(self):
        try:
            frame = int(self.time_input.text())
            self.goto_frame.emit(frame)
        except ValueError:
            cur_frame = self.viewer.dims.current_step[0]
            self.goto_frame.emit(cur_frame)
            show_warning("Time invalid, nothing changed.")

    def update_chunk_label(self):
        time_window = self.databasehandler.time_window
        label = f"window = [{time_window[0]} : {time_window[1]-1}]"
        self.chunk_label.setText(label)
