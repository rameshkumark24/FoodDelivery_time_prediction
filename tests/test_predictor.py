"""Unit tests for the serving logic around the model, using stub models."""
import logging

import joblib
import numpy as np
import pytest

from delivery.predictor import BUNDLE_FORMAT_VERSION, DeliveryTimePredictor
from delivery.validation import validate_order

# Bicycle, 10 km, 20 min prep: at most 20 km/h -> floor = 20 + 30 + 2 = 52 min.
ORDER = validate_order({"distance_km": 10, "preparation_time": 20, "vehicle": "bicycle"})
FLOOR = 52


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def make_bundle(point, lower, upper, margin=0.0, versions=None):
    return {
        "format_version": BUNDLE_FORMAT_VERSION,
        "model": ConstantModel(point),
        "lower_model": ConstantModel(lower),
        "upper_model": ConstantModel(upper),
        "interval_margin": margin,
        "metadata": {"model_name": "stub", "versions": versions or {}},
    }


def predict(point, lower, upper, margin=0.0):
    result = DeliveryTimePredictor(make_bundle(point, lower, upper, margin)).predict(ORDER)
    return result["predicted_time_min"], result["predicted_time"], result["predicted_time_max"]


def test_guardrail_raises_physically_impossible_predictions():
    assert predict(point=5, lower=3, upper=8) == (FLOOR, FLOOR, FLOOR)


def test_conformal_margin_widens_the_interval():
    assert predict(point=60, lower=55, upper=70, margin=2) == (53, 60, 72)


def test_interval_always_contains_the_prediction():
    # Independently trained quantile models can cross.
    low, point, high = predict(point=60, lower=65, upper=58)
    assert low <= point <= high


def test_breakdown_splits_kitchen_and_road_time():
    result = DeliveryTimePredictor(make_bundle(70, 60, 80)).predict(ORDER)
    assert result["breakdown"] == {"preparation_min": 20, "travel_and_handover_min": 50}


def test_unknown_bundle_format_is_rejected():
    with pytest.raises(ValueError, match="re-run 3_train_model.py"):
        DeliveryTimePredictor({"format_version": 999})


def test_library_version_mismatch_is_logged(tmp_path, caplog):
    path = tmp_path / "model.joblib"
    joblib.dump(make_bundle(60, 55, 70, versions={"scikit-learn": "0.0.1"}), path)
    with caplog.at_level(logging.WARNING, logger="delivery.predictor"):
        DeliveryTimePredictor.load(path)
    assert "trained with scikit-learn 0.0.1" in caplog.text
