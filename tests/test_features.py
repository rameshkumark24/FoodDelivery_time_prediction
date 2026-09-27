import pandas as pd
import pytest

from delivery.features import (
    FEATURE_COLUMNS,
    build_inference_frame,
    build_training_frame,
    get_time_period,
    is_peak_hour,
    minutes_between,
)

# A late-night Saturday order whose pickup crosses midnight.
RAW_ROW = {
    "Order_ID": "ORD000001",
    "Delivery_person_ID": "DEL0001",
    "Delivery_person_Age": 31,
    "Delivery_person_Ratings": 4.2,
    "Restaurant_ID": "RES0001",
    "Restaurant_latitude": 12.97,
    "Restaurant_longitude": 77.59,
    "Delivery_location_latitude": 12.99,
    "Delivery_location_longitude": 77.61,
    "Type_of_order": "Snack",
    "Type_of_vehicle": "scooter",
    "Weather_conditions": "Rainy",
    "Road_traffic_density": "Low",
    "Festival": "No",
    "City": "Metropolitan",
    "Order_Date": "2024-03-09",
    "Time_Orderd": "23:50",
    "Time_Order_picked": "00:12",
    "Distance_km": 3.07,
    "Time_taken_min": 41,
}


@pytest.mark.parametrize("hour, period", [
    (0, "Night"), (5, "Night"), (6, "Morning"), (11, "Morning"), (12, "Afternoon"),
    (16, "Afternoon"), (17, "Evening"), (20, "Evening"), (21, "Night"), (23, "Night"),
])
def test_time_period_boundaries(hour, period):
    assert get_time_period(hour) == period


@pytest.mark.parametrize("hour, expected", [
    (11, 0), (12, 1), (14, 1), (15, 0), (18, 0), (19, 1), (21, 1), (22, 0),
])
def test_peak_hours(hour, expected):
    assert is_peak_hour(hour) == expected


def test_minutes_between_wraps_past_midnight():
    start = pd.Series(["23:50", "12:00", "08:15"])
    end = pd.Series(["00:10", "12:25", "08:15"])
    # The old abs() implementation returned 1420 for the first pair.
    assert minutes_between(start, end).tolist() == [20, 25, 0]


def test_training_frame_derives_features():
    row = build_training_frame(pd.DataFrame([RAW_ROW])).iloc[0]
    assert row["Preparation_time_min"] == 22
    assert row["Order_hour"] == 23
    assert row["Day_of_week"] == 5
    assert row["Is_weekend"] == 1
    assert row["Is_peak_hour"] == 0
    assert row["Time_period"] == "Night"


def test_inference_features_match_training_features():
    """Train/serve parity: one order yields identical features via both paths."""
    trained_on = build_training_frame(pd.DataFrame([RAW_ROW]))[FEATURE_COLUMNS]
    served = build_inference_frame({
        "distance_km": RAW_ROW["Distance_km"],
        "preparation_time": 22,
        "delivery_person_age": RAW_ROW["Delivery_person_Age"],
        "delivery_person_ratings": RAW_ROW["Delivery_person_Ratings"],
        "order_hour": 23,
        "day_of_week": 5,
        "weather": RAW_ROW["Weather_conditions"],
        "traffic": RAW_ROW["Road_traffic_density"],
        "vehicle": RAW_ROW["Type_of_vehicle"],
        "order_type": RAW_ROW["Type_of_order"],
        "festival": RAW_ROW["Festival"],
        "city": RAW_ROW["City"],
    })
    assert list(served.columns) == FEATURE_COLUMNS
    pd.testing.assert_frame_equal(trained_on.reset_index(drop=True), served, check_dtype=False)
