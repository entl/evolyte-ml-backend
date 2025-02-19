import json
import os

from pydantic_settings import BaseSettings
from pydantic import Field


class Config(BaseSettings):
    ml_model_path: str = Field(os.path.join(os.path.dirname(__file__), 'ml_models', 'XGB Model.json'),
                               description='Path to the trained machine learning model.')
    scaler_path: str = Field(os.path.join(os.path.dirname(__file__), 'ml_models', 'scaler.pkl'))
    model_info_path: str = Field(os.path.join(os.path.dirname(__file__), 'ml_models', 'xgboost_model_metadata.json'))

    FEATURE_NAMES: list[str] = ['kwp', 'relative_humidity_2m', 'dew_point_2m', 'pressure_msl',
                                'precipitation', 'wind_speed_10m', 'wind_direction_10m', 'day_of_year',
                                'solar_zenith', 'solar_azimuth', 'poa', 'clearsky_index',
                                'cloud_cover_3_moving_average', 'hour_sin', 'hour_cos',
                                'day_of_year_sin', 'month_cos', 'cell_temp',
                                'physical_model_prediction']

    SCALE_FEATURES_NAMES: list[str] = ["relative_humidity_2m",
                                       "dew_point_2m",
                                       "pressure_msl",
                                       "cloud_cover_3_moving_average",
                                       "wind_speed_10m",
                                       "wind_direction_10m",
                                       "solar_zenith",
                                       "solar_azimuth",
                                       "cell_temp",
                                       ]

    LOG_SCALE_FEATURES_NAMES: list[str] = ["kwp",
                                           "poa",
                                           "physical_model_prediction",
                                           "precipitation",
                                           ]

    @property
    def model_information(self):
        """Reads model metadata from JSON file."""
        if os.path.exists(self.model_info_path):
            with open(self.model_info_path, "r") as f:
                return json.load(f)
        return {}

config = Config()
