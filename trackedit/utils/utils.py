from qtpy.QtWidgets import QTabWidget


def wrap_default_widgets_in_tabs(viewer):
    # -- 1) Identify the default dock widgets by going up the parent chain.
    # For controls: the dock widget is the direct parent.
    controls_dock = viewer.window.qt_viewer.controls.parentWidget()
    # For the layer list: go up two levels.
    list_dock = viewer.window.qt_viewer.layers.parentWidget().parentWidget()

    # -- 2) Instead of only taking the inner widget,
    # retrieve the entire container from the dock widget.
    controls_container = controls_dock.widget() if controls_dock else None
    list_container = list_dock.widget() if list_dock else None

    # -- 3) Remove the dock widgets from the main window,
    # but do not close or delete them so that Napari's internal references remain valid.
    main_window = viewer.window._qt_window
    for dock in [controls_dock, list_dock]:
        if dock is not None:
            main_window.removeDockWidget(dock)

    # -- 4) Detach the container widgets from their docks
    if controls_container:
        controls_container.setParent(None)
    if list_container:
        list_container.setParent(None)

    # -- 5) Create a tab widget and add the containers as tabs.
    tab_widget = QTabWidget()
    if controls_container:
        tab_widget.addTab(controls_container, "Layer Controls")
    if list_container:
        tab_widget.addTab(list_container, "Layer List")

    # -- 6) Add our new tab widget as a dock widget.
    new_dock = viewer.window.add_dock_widget(tab_widget, area="left", name="napari")

    # -- 7) (Optional) Update internal viewer references so that
    # Napari's menu actions refer to the new widgets.
    viewer.window.qt_viewer._controls = controls_container
    viewer.window.qt_viewer._layers = list_container

    # (Optional) Also update the internal dict of dock widgets if needed:
    # viewer.window._dock_widgets['Layer List'] = new_dock
