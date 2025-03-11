from ultrack.config import MainConfig
from ultrack.utils.test_utils import *  # Import all ultrack test utilities/fixtures

def test_opening(
    tracked_database_mock_data: MainConfig,
) -> None:
    print("opening")

    data_config = tracked_database_mock_data.data_config

    assert 1==1
