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
    """Handles standard scaling and log scaling for features."""

    def __init__(self, scaler):
        self.scaler = scaler  # StandardScaler or MinMaxScaler

    def scale(self, features: dict) -> dict:
        """Applies standard scaling to selected features."""
        features_df = pd.DataFrame([features])  # Convert to DataFrame with feature names
        scaled_df = features_df.copy()  # Copy to avoid modifying the original

        # Apply scaling only to selected features
        scaled_df[config.SCALE_FEATURES_NAMES] = self.scaler.transform(features_df[config.SCALE_FEATURES_NAMES])

        return scaled_df.iloc[0].to_dict()  # Convert back to dictionary

    def log_scale(self, features: dict) -> dict:
        """Applies log scaling to selected features."""
        log_features = np.array([[features[feature] for feature in config.LOG_SCALE_FEATURES_NAMES]])
        scaled_values = np.log1p(log_features)[0]

        for i, feature in enumerate(config.LOG_SCALE_FEATURES_NAMES):
            features[feature] = scaled_values[i]

        return features

    def process(self, features: dict) -> dict:
        """Applies both standard and log scaling where required."""
        features = self.scale(features)
        features = self.log_scale(features)
        return features


class ModelService:
    def __init__(self):
        self.model = xgboost.Booster()
        self.model.load_model(config.ml_model_path)
        scaler = joblib.load(config.scaler_path)
        self.scaler = FeatureScaler(scaler)

    def predict(self, features: PredictionRequest) -> PredictionResponse:
        """Handles prediction for a single input."""
        # Convert Pydantic model to dictionary and apply scaling
        feature_dict = features.features.dict()
        scaled_features = self.scaler.process(feature_dict)

        # Convert to DMatrix for XGBoost
        feature_values = np.array([list(scaled_features.values())])
        dmatrix = xgboost.DMatrix(feature_values, feature_names=config.FEATURE_NAMES)

        # Make prediction
        prediction = self.model.predict(dmatrix)[0]

        return PredictionResponse(prediction=str(self._ensure_non_negative(prediction)))

    def batch_predict(self, features: BatchPredictionRequest) -> BatchPredictionResponse:
        """Handles batch prediction."""
        # Convert Pydantic models to dictionaries and apply scaling
        feature_dicts = [self.scaler.process(feature.dict()) for feature in features.features]

        # Extract feature values in correct order
        feature_values = np.array([
            [feature_dict[feature] for feature in config.FEATURE_NAMES] for feature_dict in feature_dicts
        ])

        # Convert to DMatrix for XGBoost
        dmatrix = xgboost.DMatrix(feature_values, feature_names=config.FEATURE_NAMES)

        # Make batch predictions
        predictions = self.model.predict(dmatrix)
        predictions = [max(0, pred) for pred in predictions]

        return BatchPredictionResponse(
            predictions=[PredictionResponse(prediction=str(pred)) for pred in predictions]
        )

    @staticmethod
    def _ensure_non_negative(prediction: float) -> float:
        """Ensures the prediction is non-negative."""
        return max(0, prediction)
