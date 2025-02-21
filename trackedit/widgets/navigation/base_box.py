from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QSizePolicy
from qtpy.QtCore import Qt

class NavigationBox(QGroupBox):
    """Base class for navigation boxes"""
    def __init__(self, title: str, max_height: int = None):
        super().__init__()
        self.layout = QVBoxLayout()
        # title_label = QLabel(f"<h3>{title}</h3>")
        title_label = QLabel(title)
        title_label.setContentsMargins(0, 0, 0, 0)  # Remove internal margins
        title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)  # Minimize vertical space
        title_label.setStyleSheet("font-size: 14px; font-weight: bold;")  # Adjust size as needed
        self.layout.addWidget(title_label, alignment=Qt.AlignLeft)
        self.setLayout(self.layout)
        if max_height:
            self.setMaximumHeight(max_height) 