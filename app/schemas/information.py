from typing import Dict

from pydantic import BaseModel


class ModelInformationResponse(BaseModel):
    name: str
    version: str
    last_trained: str
    mae: float
    mse: float
    r2: float
    rmse: float


class FeaturesInformationResponse(BaseModel):
    """Response model providing feature names and their explanations."""
    features: Dict[str, str] = {
        "kwp": "Installed solar panel capacity in kilowatts peak (kWp).",
        "relative_humidity_2m": "Relative humidity at 2 meters above ground (%).",
        "dew_point_2m": "Dew point temperature at 2 meters above ground (°C).",
        "pressure_msl": "Atmospheric pressure at mean sea level (hPa).",
        "precipitation": "Total precipitation amount (mm).",
        "wind_speed_10m": "Wind speed at 10 meters above ground (m/s).",
        "wind_direction_10m": "Wind direction at 10 meters above ground (° from North).",
        "direct_normal_irradiance": "Direct normal solar radiation (W/m²).",
        "day_of_year": "Day of the year (1 to 365 or 366 in leap years).",
        "solar_azimuth": "Solar azimuth angle (°), indicating the sun's position.",
        "poa": "Plane of array irradiance (W/m²) on the solar panel surface.",
        "cloud_cover_3_moving_average": "Three-hour moving average of cloud cover (%).",
        "cell_temp": "Estimated solar panel cell temperature (°C).",
        "hour_sin": "Sine-transformed hour of the day to capture daily periodicity.",
        "day_of_year_sin": "Sine-transformed day of the year to capture seasonal periodicity.",
        "month_cos": "Cosine-transformed month to represent yearly cyclic patterns.",
        "relative_physical_model_prediction": "Relative prediction from a physical solar generation model."
    }