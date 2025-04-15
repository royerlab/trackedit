import sqlite3
from pathlib import Path
import sqlalchemy as sqla
from sqlalchemy import inspect, text
from ultrack.config import MainConfig
from trackedit.arrays.UltrackArray import UltrackArray
from trackedit.widgets.HierarchyWidget import HierarchyLabels, HierarchyVizWidget
import napari

def check_database(db_path: Path):
    """Check if database exists and print its structure"""
    if not db_path.exists():
        print(f"Database file does not exist at: {db_path}")
        return False
    
    print(f"Database file found at: {db_path}")
    print(f"File size: {db_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Try to connect and list tables
    engine = sqla.create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"Tables in database: {tables}")
    
    # Print column information for each table
    for table_name in tables:
        columns = inspector.get_columns(table_name)
        print(f"\nColumns in {table_name} table:")
        for column in columns:
            print(f"  - {column['name']}: {column['type']}")
            
    # Get a sample row from each table
    with engine.connect() as conn:
        for table_name in tables:
            result = conn.execute(text(f"SELECT * FROM {table_name} LIMIT 1")).fetchone()
            if result:
                print(f"\nSample row from {table_name}:")
                print(result)
    
    return True

def initialize_config():
    working_directory = Path("/mnt/md0/Teun/data/Chromatrace/2024_08_14/")
    db_filename = "data.db"
    db_path = working_directory / db_filename
    
    # if not check_database(db_path):
    #     raise FileNotFoundError(f"Database not found or invalid at {db_path}")

    # import db filename properly into an Ultrack config
    config_adjusted = MainConfig()
    config_adjusted.data_config.working_dir = working_directory
    config_adjusted.data_config.database_file_name = db_filename
    return config_adjusted

# def main():
#     config = initialize_config()
#     ultrack_array = UltrackArray(config)
    
#     labels_layer = HierarchyLabels(
#         data=ultrack_array, scale=(4,1,1), name="hierarchy"
#     ) 
#     viewer = napari.Viewer()
#     viewer.add_layer(labels_layer)
#     labels_layer.refresh()
#     labels_layer.mode = "pan_zoom"
#     napari.run()


def main2():
    config = initialize_config()
    viewer = napari.Viewer()
    hier_widget = HierarchyVizWidget(
        viewer=viewer,
        scale=(4,1,1),
        config=config,
    )
    viewer.window.add_dock_widget(hier_widget, area="bottom")
    napari.run()

if __name__ == "__main__":
    main2()
