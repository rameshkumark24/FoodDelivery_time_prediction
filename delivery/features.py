"""Feature engineering shared by training (from raw CSV rows) and serving
(from a validated API request).

Both entry points funnel through ``add_derived_features`` so a model always
sees features computed by exactly the same code it was trained with.
"""
from typing import Mapping

import numpy as np
import pandas as pd

from delivery.config import PEAK_HOURS

NUMERIC_FEATURES = [
    "Distance_km",
    "Preparation_time_min",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Order_hour",
    "Day_of_week",
    "Is_weekend",
    "Is_peak_hour",
]
CATEGORICAL_FEATURES = [
    "Weather_conditions",
    "Road_traffic_density",
    "Type_of_vehicle",
    "Type_of_order",
    "Festival",
    "City",
    "Time_period",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MINUTES_PER_DAY = 24 * 60


def get_time_period(hour: int) -> str:
    if 6 <= hour < 12:
        return "Morning"
    if 12 <= hour < 17:
        return "Afternoon"
    if 17 <= hour < 21:
        return "Evening"
    return "Night"


def is_peak_hour(hour: int) -> int:
    return int(hour in PEAK_HOURS)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from ``Order_hour`` and ``Day_of_week`` in place."""
    df["Is_weekend"] = (df["Day_of_week"] >= 5).astype(int)
    df["Is_peak_hour"] = df["Order_hour"].map(is_peak_hour).astype(int)
    df["Time_period"] = df["Order_hour"].map(get_time_period)
    return df


def minutes_between(start_hhmm: pd.Series, end_hhmm: pd.Series) -> pd.Series:
    """Minutes from ``start`` to ``end`` (HH:MM strings), wrapping past midnight.

    23:50 -> 00:10 is 20 minutes, not the 1420 minutes an ``abs()`` of the raw
    difference would give.
    """
    start = pd.to_datetime(start_hhmm, format="%H:%M")
    end = pd.to_datetime(end_hhmm, format="%H:%M")
    delta = (end - start).dt.total_seconds() / 60
    return np.mod(delta, MINUTES_PER_DAY)


def build_training_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw dataset rows (as written by ``1_generate_data.py``) into a frame
    that contains every model feature alongside the original columns."""
    df = raw.copy()
    df["Day_of_week"] = pd.to_datetime(df["Order_Date"]).dt.dayofweek
    df["Order_hour"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M").dt.hour
    df["Preparation_time_min"] = minutes_between(df["Time_Orderd"], df["Time_Order_picked"])
    return add_derived_features(df)


def build_inference_frame(order: Mapping) -> pd.DataFrame:
    """Build the single-row model input for a validated order (see
    ``delivery.validation.validate_order``)."""
    df = pd.DataFrame([{
        "Distance_km": float(order["distance_km"]),
        "Preparation_time_min": float(order["preparation_time"]),
        "Delivery_person_Age": int(order["delivery_person_age"]),
        "Delivery_person_Ratings": float(order["delivery_person_ratings"]),
        "Order_hour": int(order["order_hour"]),
        "Day_of_week": int(order["day_of_week"]),
        "Weather_conditions": order["weather"],
        "Road_traffic_density": order["traffic"],
        "Type_of_vehicle": order["vehicle"],
        "Type_of_order": order["order_type"],
        "Festival": order["festival"],
        "City": order["city"],
    }])
    return add_derived_features(df)[FEATURE_COLUMNS]
