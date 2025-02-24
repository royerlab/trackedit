import logging
import napari
import numpy as np
from typing import Sequence
from scipy import interpolate
from magicgui.widgets import Container, FloatSlider, Label
from ultrack.config import MainConfig
from trackedit.arrays.UltrackArray import UltrackArray
from qtpy.QtCore import Signal, QObject

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


class HierarchySignals(QObject):
    """Separate class to handle Qt signals"""
    click_on_hierarchy_cell = Signal(int)


class HierarchyLabels(napari.layers.Labels):
    """Extended labels layer for hierarchy visualization"""

    @property
    def _type_string(self) -> str:
        return "labels"

    def __init__(
        self,
        data: np.array,
        name: str,
        scale: tuple,
        **kwargs
    ):
        super().__init__(
            data=data,
            name=name,
            scale=scale,
            **kwargs
        )
        
        # Create signals object
        self.signals = HierarchySignals()

        # Connect click events to node selection
        @self.mouse_drag_callbacks.append
        def click(_, event):
            if event.type == "mouse_press" and self.mode == "pan_zoom" and self.visible:
                label = self.get_value(
                    event.position,
                    view_direction=event.view_direction,
                    dims_displayed=event.dims_displayed,
                    world=True
                )
                print('clicked label:', label)
                if (label is not None) and (label != 0):
                    self.signals.click_on_hierarchy_cell.emit(int(label))
                else:
                    self.signals.click_on_hierarchy_cell.emit(0)


class HierarchyVizWidget(Container):
    def __init__(
        self,
        viewer: napari.Viewer,
        scale: Sequence[float] = (1, 1, 1),
        config=None,
    ) -> None:
        """
        Initialize the HierarchyVizWidget.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer instance.
        config : MainConfig of Ultrack
            if not provided, config will be taken from UltrackWidget
        """

        super().__init__(layout="horizontal")

        self._viewer = viewer

        if config is None:
            self.config = self._get_config()
        else:
            self.config = config

        self.ultrack_array = UltrackArray(self.config)

        self.mapping = self._create_mapping()

        self._area_threshold_w = FloatSlider(label="Area", min=0, max=1, readout=False)
        self._area_threshold_w.value = 0.5
        self.ultrack_array.volume = self.mapping(0.5)
        self._area_threshold_w.changed.connect(self._slider_update)

        self.slider_label = Label(
            label=str(int(self.mapping(self._area_threshold_w.value)))
        )
        self.slider_label.native.setFixedWidth(25)

        self.append(self._area_threshold_w)
        self.append(self.slider_label)

        # Replace the standard Labels layer with our custom HierarchyLabels
        self.labels_layer = HierarchyLabels(
            data=self.ultrack_array,
            scale=scale,
            name="hierarchy"
        )
        self._viewer.add_layer(self.labels_layer)
        self.labels_layer.refresh()
        self.labels_layer.mode = 'pan_zoom'

    def _on_config_changed(self) -> None:
        self._ndim = len(self._shape)

    @property
    def _shape(self) -> Sequence[int]:
        return self.config.metadata.get("shape", [])

    def _slider_update(self, value: float) -> None:
        self.ultrack_array.volume = self.mapping(value)
        self.slider_label.label = str(int(self.mapping(value)))
        self.labels_layer.refresh()

    def _create_mapping(self):
        """
        Creates a pseudo-linear mapping from U[0,1] to full range of number of pixels
            num_pixels = mapping([0,1])
        """
        num_pixels_list = self.ultrack_array.get_tp_num_pixels(timeStart=5, timeStop=5)
        num_pixels_list.append(self.ultrack_array.minmax[0])
        num_pixels_list.append(self.ultrack_array.minmax[1])
        num_pixels_list.sort()

        x_vec = np.linspace(0, 1, len(num_pixels_list))
        y_vec = np.array(num_pixels_list)
        mapping = interpolate.interp1d(x_vec, y_vec)
        return mapping

    def _get_config(self) -> MainConfig:
        # """
        # Gets config from the Ultrack widget
        # """
        # ultrack_widget = UltrackWidget.find_ultrack_widget(self._viewer)
        # if ultrack_widget is None:
        #     raise TypeError(
        #         "config not provided and was not found within ultrack widget"
        #     )

        # return ultrack_widget._data_forms.get_config()
        pass