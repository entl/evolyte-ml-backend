from typing import Dict

from pydantic import BaseModel

class Metrics(BaseModel):
    mae: float
    mse: float
    r2: float
    rmse: float


class ModelInformationResponse(BaseModel):
    model_name: str
    model_type: str
    version: str
    metrics: Metrics
    training_date: str
