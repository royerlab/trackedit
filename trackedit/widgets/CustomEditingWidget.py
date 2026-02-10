import napari
from PyQt5.QtGui import QIntValidator, QValidator
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton

from motile_tracker.application_menus.editing_menu import EditingMenu
from trackedit.DatabaseHandler import DatabaseHandler


class CustomEditingMenu(EditingMenu):

    add_cell_button_pressed = Signal(int)
    duplicate_cell_button_pressed = Signal(int, int)
    add_spherical_cell_toggled = Signal(bool)  # Signal for spherical cell mode toggle

    def __init__(
        self,
        viewer: napari.Viewer,
        databasehandler: DatabaseHandler,
        allow_adding_spherical_cell: bool = False,
    ):
        super().__init__(viewer)  # Call the original init method
        self.databasehandler = databasehandler
        self.allow_adding_spherical_cell = allow_adding_spherical_cell

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

        # Conditionally add spherical cell button
        if self.allow_adding_spherical_cell:
            self.add_spherical_cell_btn = QPushButton("Add Spherical Cell")
            self.add_spherical_cell_btn.setCheckable(True)  # Toggle on/off
            self.add_spherical_cell_btn.setStyleSheet(
                "QPushButton:checked { background-color: #4CAF50; color: white; }"
            )
            self.add_spherical_cell_btn.clicked.connect(self._on_spherical_cell_clicked)

            spherical_cell_layout = QHBoxLayout()
            spherical_cell_layout.addWidget(self.add_spherical_cell_btn)
            spherical_cell_layout.addWidget(QLabel("R=10px"))

            node_box.layout().addLayout(spherical_cell_layout)
            node_box.setMaximumHeight(200)  # Increased to fit spherical cell button
            self.setMaximumHeight(480)  # Increased to fit spherical cell button
        else:
            node_box.setMaximumHeight(150)  # Original height
            self.setMaximumHeight(430)  # Original height

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

    def _on_spherical_cell_clicked(self, checked):
        """Emit signal when spherical cell button is toggled."""
        self.add_spherical_cell_toggled.emit(checked)
