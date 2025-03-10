import napari
from PyQt5.QtGui import QIntValidator, QValidator
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton

from motile_tracker.application_menus.editing_menu import EditingMenu
from trackedit.DatabaseHandler import DatabaseHandler


class CustomEditingMenu(EditingMenu):

    add_cell_button_pressed = Signal(int)
    duplicate_cell_button_pressed = Signal(int, int)

    def __init__(self, viewer: napari.Viewer, databasehandler: DatabaseHandler):
        super().__init__(viewer)  # Call the original init method
        self.databasehandler = databasehandler

        main_layout = self.layout()  # This retrieves the QVBoxLayout from EditingMenu
        main_layout.insertWidget(0, QLabel(r"""<h3>Edit tracks</h3>"""))

        # add cell
        self.add_cell_btn = QPushButton("Add cell")
        self.add_cell_btn.setEnabled(False)
        self.add_cell_btn.clicked.connect(self.add_cell_from_button)
        self.add_cell_input = QLineEdit()
        self.add_cell_input.setValidator(QIntValidator())
        self.add_cell_input.textChanged.connect(self.update_add_cell_btn_state)

        add_cell_layout = QHBoxLayout()
        add_cell_layout.addWidget(self.add_cell_btn)
        add_cell_layout.addWidget(self.add_cell_input)

        # duplicate cell
        self.duplicate_cell_btn = QPushButton("dupl.")
        self.duplicate_cell_btn.setEnabled(False)
        self.duplicate_cell_btn.clicked.connect(self.duplicate_cell_from_button)
        self.duplicate_cell_id_input = QLineEdit()
        self.duplicate_cell_id_input.setValidator(QIntValidator())
        self.duplicate_cell_id_input.textChanged.connect(
            self.update_duplicate_cell_btn_state
        )
        self.duplicate_time_input = QLineEdit()
        self.duplicate_time_input.setValidator(QIntValidator())
        self.duplicate_time_input.setFixedWidth(40)
        self.duplicate_time_input.textChanged.connect(
            self.update_duplicate_cell_btn_state
        )

        duplicate_cell_layout = QHBoxLayout()
        duplicate_cell_layout.addWidget(self.duplicate_cell_btn)
        duplicate_cell_layout.addWidget(self.duplicate_cell_id_input)
        duplicate_cell_layout.addWidget(QLabel("to t="))
        duplicate_cell_layout.addWidget(self.duplicate_time_input)

        # Retrieve the node_box widget from the layout and insert add/duplicate cell layouts
        node_box = main_layout.itemAt(1).widget()
        node_box.layout().addLayout(add_cell_layout)
        node_box.layout().addLayout(duplicate_cell_layout)
        node_box.setMaximumHeight(150)

        self.setMaximumHeight(430)

    def update_add_cell_btn_state(self, text):
        state, _, _ = self.add_cell_input.validator().validate(text, 0)
        self.add_cell_btn.setEnabled(state == QValidator.Acceptable)

    def update_duplicate_cell_btn_state(self, _):
        state1, _, _ = self.duplicate_cell_id_input.validator().validate(
            self.duplicate_cell_id_input.text(), 0
        )
        state2, _, _ = self.duplicate_time_input.validator().validate(
            self.duplicate_time_input.text(), 0
        )
        self.duplicate_cell_btn.setEnabled(
            state1 == QValidator.Acceptable and state2 == QValidator.Acceptable
        )

    def add_cell_from_button(self):
        node_id = int(self.add_cell_input.text())
        self.add_cell_button_pressed.emit(node_id)

    def duplicate_cell_from_button(self):
        node_id = int(self.duplicate_cell_id_input.text())
        time = int(self.duplicate_time_input.text())
        self.duplicate_cell_button_pressed.emit(node_id, time)

    def click_on_hierarchy_cell(self, label: int):
        if label > 0:
            self.add_cell_input.setText(str(label))
            self.duplicate_cell_id_input.setText(str(label))
        else:
            self.add_cell_input.setText("")
            self.duplicate_cell_id_input.setText("")
