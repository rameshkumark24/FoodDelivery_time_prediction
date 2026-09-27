from datetime import datetime

import pytest

from delivery.validation import ValidationError, validate_order

NOW = datetime(2024, 5, 17, 19, 30)  # a Friday evening


def test_optional_fields_take_defaults():
    assert validate_order({"distance_km": 3}, now=NOW) == {
        "distance_km": 3.0,
        "preparation_time": 15,
        "delivery_person_age": 30,
        "delivery_person_ratings": 4.5,
        "order_hour": 19,
        "day_of_week": 4,
        "weather": "Sunny",
        "traffic": "Medium",
        "vehicle": "motorcycle",
        "order_type": "Meal",
        "festival": "No",
        "city": "Urban",
    }


def test_explicit_values_override_clock_defaults():
    order = validate_order({"distance_km": 3, "order_hour": 0, "day_of_week": 0}, now=NOW)
    assert (order["order_hour"], order["day_of_week"]) == (0, 0)


@pytest.mark.parametrize("field, value, expected", [
    ("weather", "stormy", "Stormy"),
    ("traffic", "JAM", "Jam"),
    ("vehicle", "Electric Scooter", "electric_scooter"),
    ("vehicle", "electric-scooter", "electric_scooter"),
    ("city", "semi urban", "Semi-Urban"),
    ("order_type", " buffet ", "Buffet"),
    ("festival", True, "Yes"),
    ("festival", False, "No"),
])
def test_categories_are_normalised(field, value, expected):
    assert validate_order({"distance_km": 3, field: value}, now=NOW)[field] == expected


def test_numeric_strings_are_accepted():
    order = validate_order({"distance_km": "4.5", "delivery_person_age": "31"}, now=NOW)
    assert order["distance_km"] == 4.5
    assert order["delivery_person_age"] == 31


@pytest.mark.parametrize("payload, field", [
    ({}, "distance_km"),
    ({"distance_km": None}, "distance_km"),
    ({"distance_km": -1}, "distance_km"),
    ({"distance_km": 0}, "distance_km"),
    ({"distance_km": 30}, "distance_km"),
    ({"distance_km": "far"}, "distance_km"),
    ({"distance_km": float("nan")}, "distance_km"),
    ({"distance_km": float("inf")}, "distance_km"),
    ({"distance_km": 10 ** 400}, "distance_km"),
    ({"distance_km": True}, "distance_km"),
    ({"distance_km": 3, "preparation_time": 0}, "preparation_time"),
    ({"distance_km": 3, "delivery_person_age": 30.5}, "delivery_person_age"),
    ({"distance_km": 3, "delivery_person_age": 17}, "delivery_person_age"),
    ({"distance_km": 3, "delivery_person_ratings": 5.5}, "delivery_person_ratings"),
    ({"distance_km": 3, "order_hour": 24}, "order_hour"),
    ({"distance_km": 3, "day_of_week": 7}, "day_of_week"),
    ({"distance_km": 3, "weather": "Snow"}, "weather"),
    ({"distance_km": 3, "vehicle": 1}, "vehicle"),
    ({"distance_km": 3, "festival": "maybe"}, "festival"),
])
def test_invalid_fields_are_rejected(payload, field):
    with pytest.raises(ValidationError) as excinfo:
        validate_order(payload, now=NOW)
    assert field in excinfo.value.errors


def test_all_errors_are_reported_together():
    with pytest.raises(ValidationError) as excinfo:
        validate_order({"distance_km": -1, "weather": "Snow", "order_hour": 99}, now=NOW)
    assert set(excinfo.value.errors) == {"distance_km", "weather", "order_hour"}


@pytest.mark.parametrize("payload", [None, [], "text", 5])
def test_non_object_payloads_are_rejected(payload):
    with pytest.raises(ValidationError) as excinfo:
        validate_order(payload, now=NOW)
    assert "body" in excinfo.value.errors
