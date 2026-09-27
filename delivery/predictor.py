"""Load the trained model bundle and turn validated orders into predictions."""
import logging
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import joblib
import pandas as pd

from delivery.config import (
    INTERVAL_LOWER_QUANTILE,
    INTERVAL_UPPER_QUANTILE,
    MIN_HANDOVER_MIN,
    VEHICLE_MAX_SPEED_KMH,
)
from delivery.features import build_inference_frame

logger = logging.getLogger(__name__)

BUNDLE_FORMAT_VERSION = 1

# Package -> the distributions that can provide it (the ``xgboost-cpu`` wheel
# installs the ``xgboost`` package).
TRACKED_PACKAGES = {
    "scikit-learn": ("scikit-learn",),
    "xgboost": ("xgboost", "xgboost-cpu"),
    "lightgbm": ("lightgbm",),
    "numpy": ("numpy",),
    "pandas": ("pandas",),
    "joblib": ("joblib",),
}


def installed_versions() -> Dict[str, str]:
    """Versions from package metadata, without importing the packages."""
    versions = {}
    for name, distributions in TRACKED_PACKAGES.items():
        versions[name] = "not installed"
        for distribution in distributions:
            try:
                versions[name] = metadata.version(distribution)
                break
            except metadata.PackageNotFoundError:
                continue
    return versions


def minimum_feasible_minutes(distance_km, preparation_time, vehicle):
    """Lower bound on total delivery time: the food must be prepared, then
    ridden at no more than the vehicle's top urban speed, then handed over.

    Works on scalars (serving) and on arrays/Series (evaluation in training).
    """
    if isinstance(vehicle, str):
        top_speed = VEHICLE_MAX_SPEED_KMH[vehicle]
    else:
        top_speed = pd.Series(vehicle).map(VEHICLE_MAX_SPEED_KMH).to_numpy()
    return preparation_time + distance_km / top_speed * 60 + MIN_HANDOVER_MIN


class DeliveryTimePredictor:
    def __init__(self, bundle: Mapping[str, Any]):
        if bundle.get("format_version") != BUNDLE_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported model bundle format {bundle.get('format_version')!r}; "
                "re-run 3_train_model.py"
            )
        self.model = bundle["model"]
        self.lower_model = bundle["lower_model"]
        self.upper_model = bundle["upper_model"]
        # Conformal calibration: widen both interval ends by this many minutes.
        self.interval_margin = float(bundle["interval_margin"])
        self.metadata = bundle["metadata"]

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DeliveryTimePredictor":
        predictor = cls(joblib.load(path))
        predictor._warn_on_version_mismatch()
        return predictor

    @property
    def model_name(self) -> str:
        return self.metadata["model_name"]

    def _warn_on_version_mismatch(self) -> None:
        trained_with = self.metadata.get("versions", {})
        running = installed_versions()
        for package, version in trained_with.items():
            if running.get(package) != version:
                logger.warning(
                    "Model was trained with %s %s but %s is installed; "
                    "predictions may be unreliable. Install requirements.txt "
                    "or retrain.", package, version, running.get(package),
                )

    def predict(self, order: Mapping[str, Any]) -> Dict[str, Any]:
        """Predict for an order already normalised by ``validate_order``."""
        X = build_inference_frame(order)
        raw_point = float(self.model.predict(X)[0])
        floor = minimum_feasible_minutes(
            order["distance_km"], order["preparation_time"], order["vehicle"])
        point = max(raw_point, floor)
        if raw_point < floor:
            logger.info("Guardrail raised prediction %.1f -> %.1f min", raw_point, floor)

        lower = float(self.lower_model.predict(X)[0]) - self.interval_margin
        upper = float(self.upper_model.predict(X)[0]) + self.interval_margin

        predicted = round(point)
        low = min(round(max(lower, floor)), predicted)
        high = max(round(upper), predicted)
        prep = order["preparation_time"]

        return {
            "predicted_time": predicted,
            "predicted_time_min": low,
            "predicted_time_max": high,
            "interval_confidence": round(INTERVAL_UPPER_QUANTILE - INTERVAL_LOWER_QUANTILE, 2),
            "breakdown": {
                "preparation_min": round(prep, 1),
                "travel_and_handover_min": round(max(predicted - prep, 0), 1),
            },
        }
