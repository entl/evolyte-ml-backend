from fastapi import APIRouter, Depends, status, HTTPException
from app.schemas.prediction import PredictionRequest, PredictionResponse, BatchPredictionRequest, \
    BatchPredictionResponse
from app.services.model_service import ModelService

# Define router for predictions
predict_router = APIRouter(prefix="/prediction", tags=["Predictions"])


@predict_router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictionRequest, model_service: ModelService = Depends()):
    """Handles single prediction requests."""
    try:
        prediction = model_service.predict(request)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@predict_router.post("/batch_predict", response_model=BatchPredictionResponse, status_code=status.HTTP_200_OK)
def batch_predict(request: BatchPredictionRequest, model_service: ModelService = Depends()):
    """Handles batch prediction requests."""
    try:
        predictions = model_service.batch_predict(request)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")
