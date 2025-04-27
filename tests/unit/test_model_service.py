import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
from datetime import datetime

import xgboost
import joblib

from app.services.model_service import MLModelService, FeatureScaler
import app.config as config

# config uses the full feature sets from schema
FEATURE_NAMES = ['kwp', 'relative_humidity_2m', 'dew_point_2m', 'pressure_msl',
                 'precipitation', 'wind_speed_10m', 'wind_direction_10m', 'day_of_year',
                 'solar_zenith', 'solar_azimuth', 'poa', 'clearsky_index',
                 'cloud_cover_3_moving_average', 'hour_sin', 'hour_cos',
                 'day_of_year_sin', 'month_cos', 'cell_temp', 'physical_model_prediction']
config.FEATURE_NAMES = FEATURE_NAMES
config.SCALE_FEATURES_NAMES = [
    "relative_humidity_2m", "dew_point_2m", "pressure_msl",
    "cloud_cover_3_moving_average", "wind_speed_10m",
    "wind_direction_10m", "solar_zenith", "solar_azimuth", "cell_temp"
]
config.LOG_SCALE_FEATURES_NAMES = [
    "kwp", "poa", "physical_model_prediction", "precipitation"
]
config.ml_model_path = 'dummy_model_path'
config.scaler_path = 'dummy_scaler_path'


def base_features_dict(value=1.0):
    """Helper to create a full feature dict with all FEATURE_NAMES and with dummy value"""
    return {name: value for name in config.FEATURE_NAMES}


def test_feature_scaler_scale():
    # full features dict
    features = base_features_dict(1.0)

    # transform returns doubled values for scale features (mimics scaler)
    mock_scaler = MagicMock()
    num_scaled = len(config.SCALE_FEATURES_NAMES)
    mock_scaler.transform.return_value = np.ones((1, num_scaled)) * 2
    scaler = FeatureScaler(mock_scaler)

    scaled = scaler.scale(features.copy())

    # as per rule defined scaler doubles values
    for key in config.SCALE_FEATURES_NAMES:
        assert scaled[key] == pytest.approx(2.0)
    # all other features should be unchanged
    for key in set(config.FEATURE_NAMES) - set(config.SCALE_FEATURES_NAMES):
        assert scaled[key] == pytest.approx(1.0)

    # Ensure transform was called with the correct frame
    called_arg = mock_scaler.transform.call_args[0][0]

    assert list(called_arg.columns) == config.SCALE_FEATURES_NAMES
    assert all(called_arg.iloc[0] == 1.0)

@pytest.mark.parametrize("input_value,expected", [
    (np.e - 1, np.log1p(np.e - 1)),
    (0.0, 0.0)
])
def test_feature_scaler_log_scale(input_value, expected):
    # init features for log scale
    features = base_features_dict(input_value)

    scaler = FeatureScaler(None)
    result = scaler.log_scale(features.copy())

    # Each log-scaled feature should match
    for key in config.LOG_SCALE_FEATURES_NAMES:
        assert pytest.approx(expected) == result[key]


def test_feature_scaler_process_calls_scale_and_log_scale(monkeypatch):
    scaler = MagicMock()
    fs = FeatureScaler(scaler)

    # fake returns from scaler
    monkeypatch.setattr(fs, 'scale', MagicMock(return_value={'f': 3}))
    monkeypatch.setattr(fs, 'log_scale', MagicMock(return_value={'f': 42}))

    processed = fs.process({'f': 1})
    fs.scale.assert_called_once()
    fs.log_scale.assert_called_once()
    assert processed['f'] == 42

@patch('app.services.model_service.xgboost.Booster')
@patch('app.services.model_service.joblib.load')
def test_predict(mock_joblib_load, mock_booster_cls):
    mock_scaler = MagicMock()
    # scale returns ones, so after log_scale only kwp change but dummy log not applied here
    mock_scaler.transform.return_value = np.ones((1, len(config.SCALE_FEATURES_NAMES)))
    mock_joblib_load.return_value = mock_scaler

    model_instance = MagicMock()
    model_instance.predict.return_value = np.array([0.5])
    mock_booster_cls.return_value = model_instance

    service = MLModelService()

    # Dummy features class returns full dict and has kwp
    class DummyFeatures:
        def __init__(self, kwp):
            self.kwp = kwp
        def dict(self):
            return base_features_dict(1.0)

    request = SimpleNamespace(
        features=DummyFeatures(kwp=10.0),
        datetime=datetime(2025, 4, 22, 12, 0)
    )
    response = service.predict(request)

    # should call transform and predict
    mock_scaler.transform.assert_called_once()
    model_instance.predict.assert_called_once()

    # 0.5 * kwp = 5.0
    # since prediction converts relative output to kwh
    assert response.prediction == pytest.approx(5.0)
    assert response.datetime == request.datetime

@patch('app.services.model_service.xgboost.Booster')
@patch('app.services.model_service.joblib.load')
def test_predict_negative(mock_joblib_load, mock_booster_cls):
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.ones((1, len(config.SCALE_FEATURES_NAMES)))
    mock_joblib_load.return_value = mock_scaler

    model_instance = MagicMock()
    model_instance.predict.return_value = np.array([-0.2])
    mock_booster_cls.return_value = model_instance

    service = MLModelService()

    class DummyFeatures:
        def __init__(self, kwp):
            self.kwp = kwp
        def dict(self):
            return base_features_dict(1.0)

    request = SimpleNamespace(
        features=DummyFeatures(kwp=5.0),
        datetime=datetime(2025, 4, 22, 15, 0)
    )
    response = service.predict(request)

    # Negative outputs should floor to zero
    assert response.prediction == pytest.approx(0.0)

@patch('app.services.model_service.xgboost.Booster')
@patch('app.services.model_service.joblib.load')
def test_batch_predict(mock_joblib_load, mock_booster_cls):
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = np.ones((1, len(config.SCALE_FEATURES_NAMES)))
    mock_joblib_load.return_value = mock_scaler

    model_instance = MagicMock()
    model_instance.predict.return_value = np.array([0.5])
    mock_booster_cls.return_value = model_instance

    service = MLModelService()

    class DummyFeatures:
        def __init__(self, kwp):
            self.kwp = kwp
        def dict(self):
            return base_features_dict(1.0)

    entries = [
        SimpleNamespace(features=DummyFeatures(kwp=2.0), datetime=datetime(2025, 4, 22, 9, 0)),
        SimpleNamespace(features=DummyFeatures(kwp=4.0), datetime=datetime(2025, 4, 22, 10, 0))
    ]
    batch_request = SimpleNamespace(entries=entries)
    batch_response = service.batch_predict(batch_request)

    assert len(batch_response.predictions) == 2
    # 0.5 * kwp
    # same as in test_predict
    assert batch_response.predictions[0].prediction == pytest.approx(1.0)
    assert batch_response.predictions[1].prediction == pytest.approx(2.0)
