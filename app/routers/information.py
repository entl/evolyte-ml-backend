from fastapi import APIRouter, status

from app.schemas.information import ModelInformationResponse
from app.config import config

information_router = APIRouter()


@information_router.get("/model-info", status_code=status.HTTP_200_OK)
def model_info():
    return ModelInformationResponse(**config.model_information)

