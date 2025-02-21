from qtpy.QtWidgets import QLabel
from qtpy.QtCore import Signal

class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit()