import joblib
import numpy as np
import xgboost
import pandas as pd

from app.schemas.prediction import (
    PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse
)
from app.config import config


class FeatureScaler:
    def __init__(self, scaler):
        self.scaler = scaler

    def scale(self, features: dict) -> dict:
        features_df = pd.DataFrame([features])
        scaled_df = features_df.copy()

        # apply scaling only to selected features
        scaled_df[config.SCALE_FEATURES_NAMES] = self.scaler.transform(features_df[config.SCALE_FEATURES_NAMES])

        return scaled_df.iloc[0].to_dict()

    def log_scale(self, features: dict) -> dict:
        log_features = np.array([[features[feature] for feature in config.LOG_SCALE_FEATURES_NAMES]])
        scaled_values = np.log1p(log_features)[0]

        for i, feature in enumerate(config.LOG_SCALE_FEATURES_NAMES):
            features[feature] = scaled_values[i]

        return features

    def process(self, features: dict) -> dict:
        features = self.scale(features)
        features = self.log_scale(features)
        return features


class MLModelService:
    def __init__(self):
        self.model = xgboost.Booster()
        self.model.load_model(config.ml_model_path)
        scaler = joblib.load(config.scaler_path)
        self.scaler = FeatureScaler(scaler)

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        # convert Pydantic model to dictionary and apply scaling
        feature_dict = request.features.dict()
        scaled_features = self.scaler.process(feature_dict)

        # convert to DMatrix for XGBoost
        feature_values = np.array([list(scaled_features.values())])
        dmatrix = xgboost.DMatrix(feature_values, feature_names=config.FEATURE_NAMES)

        prediction = self.model.predict(dmatrix)[0]
        prediction = self._ensure_non_negative(prediction)

        prediction = self._convert_relative_output_to_kwh(request.features.kwp, prediction)

        return PredictionResponse(prediction=prediction,
                                  datetime=request.datetime)

    def batch_predict(self, request: BatchPredictionRequest):
        entries = request.entries
        predictions = []

        for entry in entries:
            feature_dict = entry.features.dict()
            scaled_features = self.scaler.process(feature_dict)

            # Convert to DMatrix for XGBoost
            feature_values = np.array([list(scaled_features.values())])
            dmatrix = xgboost.DMatrix(feature_values, feature_names=config.FEATURE_NAMES)

            prediction = self.model.predict(dmatrix)[0]

            prediction = self._ensure_non_negative(prediction)
            prediction = self._convert_relative_output_to_kwh(entry.features.kwp, prediction)

            predictions.append(PredictionResponse(prediction=str(prediction), datetime=entry.datetime))

        return BatchPredictionResponse(predictions=predictions)

    @staticmethod
    def _ensure_non_negative(prediction: float) -> float:
        # ensure the prediction is non-negative
        return max(0, prediction)

    @staticmethod
    def _convert_relative_output_to_kwh(kwp: float, relative_output: float) -> float:
        return kwp * relative_output
