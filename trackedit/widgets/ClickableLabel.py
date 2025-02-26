from qtpy.QtCore import Signal
from qtpy.QtWidgets import QLabel


class ClickableLabel(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.clicked.emit()
