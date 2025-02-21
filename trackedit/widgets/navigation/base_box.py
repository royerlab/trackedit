from qtpy.QtWidgets import QGroupBox, QVBoxLayout, QLabel, QSizePolicy
from qtpy.QtCore import Qt

class NavigationBox(QGroupBox):
    """Base class for navigation boxes"""
    def __init__(self, title: str, max_height: int = None):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(5, 2, 5, 2)  # Reduce the margins inside the group box
        self.layout.setSpacing(2)  # Reduce spacing between elements
        
        title_label = QLabel(title)
        title_label.setContentsMargins(0, 0, 0, 0)
        title_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.layout.addWidget(title_label, alignment=Qt.AlignLeft)
        
        # Set group box properties to minimize spacing
        self.setFlat(True)  # Makes the group box border less prominent
        self.setContentsMargins(2, 2, 2, 2)  # Minimal margins around the group box
        
        if max_height:
            self.setMaximumHeight(max_height)
            
        self.setLayout(self.layout) 