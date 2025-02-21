from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QLabel
from qtpy.QtCore import Qt

class NavigationBox(QGroupBox):
    """Base class for navigation boxes"""
    def __init__(self, title: str, max_height: int = None):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel(f"<h3>{title}</h3>"), alignment=Qt.AlignLeft)
        self.setLayout(self.layout)
        if max_height:
            self.setMaximumHeight(max_height) 