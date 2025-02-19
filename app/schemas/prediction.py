from pydantic import BaseModel
from typing import List


# Define the expected features as a dedicated model.
class FeatureInput(BaseModel):
    kwp: float
    relative_humidity_2m: float
    dew_point_2m: float
    pressure_msl: float
    precipitation: float
    wind_speed_10m: float
    wind_direction_10m: float
    direct_normal_irradiance: float
    day_of_year: float
    solar_azimuth: float
    poa: float
    cloud_cover_3_moving_average: float
    cell_temp: float
    hour_sin: float
    day_of_year_sin: float
    month_cos: float
    relative_physical_model_prediction: float


# Request model for a single prediction.
class PredictionRequest(BaseModel):
    features: FeatureInput


# Response model for a single prediction.
class PredictionResponse(BaseModel):
    prediction: str


# Request model for batch predictions.
class BatchPredictionRequest(BaseModel):
    features: List[FeatureInput]


# Response model for batch predictions.
class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
