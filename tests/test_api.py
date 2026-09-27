"""API tests against the committed model bundle (so they also prove the
bundle loads with the pinned library versions)."""
import pytest

from app import create_app
from delivery.predictor import minimum_feasible_minutes


def predict(client, payload):
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.get_json()
    return response.get_json()


def test_home_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "GourmetExpress" in html
    assert 'id="distance_km"' in html
    assert "Model performance" in html


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["model_loaded"] is True


def test_model_info_reports_a_useful_model(client):
    info = client.get("/api/model-info").get_json()
    assert info["model_name"] != "Baseline (median)"
    assert info["test_metrics"]["mae"] < 5
    assert info["test_metrics"]["r2"] > 0.9
    assert 75 <= info["prediction_interval"]["test_coverage_pct"] <= 90
    assert "Distance_km" in info["feature_importances"]


def test_predict_response(client, order):
    data = predict(client, order)
    assert data["success"] is True
    assert data["predicted_time_min"] <= data["predicted_time"] <= data["predicted_time_max"]
    assert data["interval_confidence"] == 0.8
    breakdown = data["breakdown"]
    assert breakdown["preparation_min"] == order["preparation_time"]
    assert breakdown["preparation_min"] + breakdown["travel_and_handover_min"] == data["predicted_time"]
    assert data["message"] == f"Estimated delivery time: {data['predicted_time']} minutes"
    assert data["inputs"] == order


def test_minimal_payload_uses_defaults(client):
    data = predict(client, {"distance_km": 4})
    assert data["inputs"]["weather"] == "Sunny"
    assert data["inputs"]["vehicle"] == "motorcycle"


def test_prediction_grows_with_distance(client, order):
    times = [predict(client, {**order, "distance_km": d})["predicted_time"]
             for d in (1, 3, 6, 10, 15, 20, 25)]
    assert times == sorted(set(times)), times


@pytest.mark.parametrize("field, faster, slower", [
    ("preparation_time", 10, 40),
    ("traffic", "Low", "Jam"),
    ("weather", "Sunny", "Stormy"),
    ("vehicle", "motorcycle", "bicycle"),
    ("city", "Semi-Urban", "Metropolitan"),
])
def test_prediction_responds_to_conditions(client, order, field, faster, slower):
    fast = predict(client, {**order, field: faster})["predicted_time"]
    slow = predict(client, {**order, field: slower})["predicted_time"]
    assert fast < slow


@pytest.mark.parametrize("vehicle", ["motorcycle", "bicycle"])
@pytest.mark.parametrize("distance_km", [0.1, 12, 25])
@pytest.mark.parametrize("preparation_time", [1, 60])
def test_prediction_is_physically_feasible(client, order, vehicle, distance_km, preparation_time):
    payload = {**order, "vehicle": vehicle, "distance_km": distance_km,
               "preparation_time": preparation_time, "traffic": "Low"}
    data = predict(client, payload)
    floor = minimum_feasible_minutes(distance_km, preparation_time, vehicle)
    assert data["predicted_time"] >= round(floor)
    assert data["predicted_time_min"] <= data["predicted_time"] <= data["predicted_time_max"]


def test_invalid_input_returns_field_errors(client, order):
    response = client.post("/predict", json={**order, "distance_km": -5, "weather": "Snow"})
    assert response.status_code == 400
    data = response.get_json()
    assert data["success"] is False
    assert set(data["errors"]) == {"distance_km", "weather"}


@pytest.mark.parametrize("kwargs", [
    {"data": "distance=5", "content_type": "text/plain"},
    {"data": "{not json", "content_type": "application/json"},
])
def test_non_json_body_is_rejected(client, kwargs):
    response = client.post("/predict", **kwargs)
    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_huge_integer_is_a_validation_error_not_a_crash(client):
    body = '{"distance_km": 1' + "0" * 400 + "}"
    response = client.post("/predict", data=body, content_type="application/json")
    assert response.status_code == 400
    assert "distance_km" in response.get_json()["errors"]


def test_oversized_body_is_rejected(client):
    body = '{"distance_km": 5, "padding": "' + "x" * 20_000 + '"}'
    response = client.post("/predict", data=body, content_type="application/json")
    assert response.status_code == 413


def test_unknown_route_returns_json_404(client):
    response = client.get("/does-not-exist")
    assert response.status_code == 404
    assert response.get_json()["success"] is False


def test_missing_model_is_reported_not_hidden(tmp_path, order):
    client = create_app(model_path=tmp_path / "missing.joblib").test_client()
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=order).status_code == 503
    assert client.get("/api/model-info").status_code == 503
    page = client.get("/")
    assert page.status_code == 200
    assert "model is not loaded" in page.get_data(as_text=True)
