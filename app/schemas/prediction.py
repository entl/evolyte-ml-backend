from pydantic import BaseModel
from typing import List

from datetime import datetime


class FeatureInput(BaseModel):
    kwp: float
    relative_humidity_2m: float
    dew_point_2m: float
    pressure_msl: float
    precipitation: float
    wind_speed_10m: float
    wind_direction_10m: float
    day_of_year: float
    solar_zenith: float
    solar_azimuth: float
    poa: float
    clearsky_index: float
    cloud_cover_3_moving_average: float
    hour_sin: float
    hour_cos: float
    day_of_year_sin: float
    month_cos: float
    cell_temp: float
    physical_model_prediction: float


class PredictionRequest(BaseModel):
    features: FeatureInput
    datetime: datetime


class PredictionResponse(BaseModel):
    prediction: float
    datetime: datetime


class BatchPredictionRequest(BaseModel):
    entries: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
