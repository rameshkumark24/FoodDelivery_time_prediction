"""Project-wide constants shared by data generation, training and serving."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
VIZ_DIR = ROOT_DIR / "visualizations"

RAW_DATA_PATH = DATA_DIR / "food_delivery_data.csv"
SAMPLE_DATA_PATH = DATA_DIR / "food_delivery_sample.csv"
PROCESSED_DATA_PATH = DATA_DIR / "food_delivery_processed.csv"
MODEL_PATH = MODELS_DIR / "delivery_model.joblib"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"

RANDOM_STATE = 42
TARGET = "Time_taken_min"

# ----------------------------------------------------------------------------
# Categorical levels (canonical spelling used everywhere)
# ----------------------------------------------------------------------------
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Fog", "Stormy"]
TRAFFIC_LEVELS = ["Low", "Medium", "High", "Jam"]
VEHICLE_TYPES = ["motorcycle", "scooter", "electric_scooter", "bicycle"]
ORDER_TYPES = ["Snack", "Meal", "Drinks", "Buffet"]
CITY_TYPES = ["Urban", "Semi-Urban", "Metropolitan"]
FESTIVAL_OPTIONS = ["No", "Yes"]
TIME_PERIODS = ["Morning", "Afternoon", "Evening", "Night"]
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# Human-friendly labels for the web form.
DISPLAY_LABELS = {
    "Sunny": "Sunny ☀️", "Cloudy": "Cloudy ☁️", "Rainy": "Rainy 🌧️",
    "Fog": "Fog 🌫️", "Stormy": "Stormy ⛈️",
    "Low": "Low 🟢", "Medium": "Medium 🟡", "High": "High 🟠", "Jam": "Jam 🔴",
    "motorcycle": "Motorcycle 🏍️", "scooter": "Scooter 🛵",
    "electric_scooter": "Electric scooter ⚡", "bicycle": "Bicycle 🚲",
    "Snack": "Snack 🥨", "Meal": "Meal 🍔", "Drinks": "Drinks 🥤", "Buffet": "Buffet 🍱",
    "Urban": "Urban 🏙️", "Semi-Urban": "Semi-Urban 🏡", "Metropolitan": "Metropolitan 🌆",
    "No": "No", "Yes": "Yes 🎉",
}

# Lunch (12-14h) and dinner (19-21h) rush, inclusive.
PEAK_HOURS = frozenset({12, 13, 14, 19, 20, 21})

# ----------------------------------------------------------------------------
# API input specification (also the ranges covered by the training data)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class NumericSpec:
    label: str
    min: float
    max: float
    default: Optional[float]  # None -> required (or derived from the clock)
    integer: bool = False
    step: float = 1


NUMERIC_INPUTS = {
    "distance_km": NumericSpec("Distance (km)", 0.1, 25.0, None, step=0.1),
    "preparation_time": NumericSpec("Preparation time (min)", 1, 60, 15),
    "delivery_person_age": NumericSpec("Rider age", 18, 60, 30, integer=True),
    "delivery_person_ratings": NumericSpec("Rider rating", 1.0, 5.0, 4.5, step=0.1),
    "order_hour": NumericSpec("Order hour", 0, 23, None, integer=True),
    "day_of_week": NumericSpec("Day of week", 0, 6, None, integer=True),
}

# Fields that default to the server clock when omitted.
CLOCK_FIELDS = ("order_hour", "day_of_week")

CATEGORICAL_INPUTS = {
    "weather": (WEATHER_CONDITIONS, "Sunny"),
    "traffic": (TRAFFIC_LEVELS, "Medium"),
    "vehicle": (VEHICLE_TYPES, "motorcycle"),
    "order_type": (ORDER_TYPES, "Meal"),
    "festival": (FESTIVAL_OPTIONS, "No"),
    "city": (CITY_TYPES, "Urban"),
}

# ----------------------------------------------------------------------------
# Serving guardrail: no delivery can beat the vehicle's top urban speed.
# ----------------------------------------------------------------------------
VEHICLE_MAX_SPEED_KMH = {
    "motorcycle": 45,
    "scooter": 40,
    "electric_scooter": 30,
    "bicycle": 20,
}
MIN_HANDOVER_MIN = 2  # pickup + drop-off can't be instantaneous

# Width of the prediction interval shown to users.
INTERVAL_LOWER_QUANTILE = 0.1
INTERVAL_UPPER_QUANTILE = 0.9
