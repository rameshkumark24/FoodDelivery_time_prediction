"""Flask web app and JSON API for food delivery time predictions."""
import os

# LightGBM and scikit-learn use OpenMP, whose thread pool does not survive
# fork(): under ``gunicorn --preload`` a worker would deadlock on its first
# prediction. Single-row predictions gain nothing from threads anyway. This
# must run before any OpenMP-backed library is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import logging  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Union  # noqa: E402

from flask import Flask, jsonify, render_template, request  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

from delivery.config import (  # noqa: E402
    CATEGORICAL_INPUTS,
    DAY_NAMES,
    DISPLAY_LABELS,
    MODEL_PATH,
    NUMERIC_INPUTS,
)
from delivery.predictor import DeliveryTimePredictor  # noqa: E402
from delivery.validation import ValidationError, validate_order  # noqa: E402

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("delivery.app")


def load_predictor(path: Union[str, Path]) -> Optional[DeliveryTimePredictor]:
    try:
        predictor = DeliveryTimePredictor.load(path)
    except FileNotFoundError:
        logger.error("Model file %s not found - run 3_train_model.py first", path)
        return None
    except Exception:
        logger.exception("Could not load model from %s - run 3_train_model.py first", path)
        return None
    logger.info("Loaded model %s from %s", predictor.model_name, path)
    return predictor


def form_options():
    """Everything the page needs to render inputs that match the API rules."""
    return {
        "numeric": NUMERIC_INPUTS,
        "categorical": {
            name: {"default": default,
                   "choices": [(level, DISPLAY_LABELS.get(level, level)) for level in levels]}
            for name, (levels, default) in CATEGORICAL_INPUTS.items()
        },
        "days": list(enumerate(DAY_NAMES)),
    }


def public_model_info(predictor: DeliveryTimePredictor) -> dict:
    meta = predictor.metadata
    return {
        "model_name": meta["model_name"],
        "trained_at": meta["trained_at"],
        "training_samples": meta["n_train"],
        "test_samples": meta["n_test"],
        "test_metrics": meta["test_metrics"],
        "cross_validation": {"folds": meta["cv_folds"], **meta["cv"]},
        "prediction_interval": meta["interval"],
        "feature_importances": meta["feature_importances"],
        "model_comparison": meta["comparison"],
    }


def create_app(model_path: Union[str, Path, None] = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # requests are tiny JSON objects
    app.json.sort_keys = False
    predictor = load_predictor(model_path or os.environ.get("MODEL_PATH", MODEL_PATH))

    def model_unavailable():
        return jsonify(success=False, error="Model is not loaded; the service is unavailable."), 503

    @app.get("/")
    def home():
        info = public_model_info(predictor) if predictor else None
        return render_template("index.html", model_info=info, form=form_options())

    @app.post("/predict")
    def predict():
        if predictor is None:
            return model_unavailable()
        payload = request.get_json(silent=True)
        if payload is None:
            return jsonify(success=False, error="Request body must be valid JSON "
                                                "sent with Content-Type: application/json."), 400
        try:
            order = validate_order(payload)
        except ValidationError as exc:
            return jsonify(success=False, error="Invalid input.", errors=exc.errors), 400

        result = predictor.predict(order)
        return jsonify(
            success=True,
            **result,
            message=f"Estimated delivery time: {result['predicted_time']} minutes",
            model=predictor.model_name,
            inputs=order,
        )

    @app.get("/api/model-info")
    def model_info():
        if predictor is None:
            return model_unavailable()
        return jsonify(public_model_info(predictor))

    @app.get("/health")
    def health():
        status = 200 if predictor else 503
        return jsonify(status="healthy" if predictor else "unhealthy",
                       model_loaded=predictor is not None,
                       model=predictor.model_name if predictor else None), status

    @app.errorhandler(HTTPException)
    def http_error(exc: HTTPException):
        return jsonify(success=False, error=exc.description), exc.code

    @app.errorhandler(Exception)
    def unexpected_error(exc: Exception):
        logger.exception("Unhandled error on %s", request.path)
        return jsonify(success=False, error="Internal server error."), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
