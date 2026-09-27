import pytest

from app import create_app


@pytest.fixture(scope="session")
def client():
    return create_app().test_client()


@pytest.fixture
def order():
    """A complete, valid /predict payload."""
    return {
        "distance_km": 5.0,
        "preparation_time": 15,
        "delivery_person_age": 28,
        "delivery_person_ratings": 4.5,
        "order_hour": 13,
        "day_of_week": 2,
        "weather": "Sunny",
        "traffic": "Medium",
        "vehicle": "motorcycle",
        "order_type": "Meal",
        "festival": "No",
        "city": "Urban",
    }
