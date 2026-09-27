"""End-to-end checks of the data -> EDA -> training pipeline on a small dataset."""
import importlib

import numpy as np
import pandas as pd
import pytest

from delivery.config import (
    CITY_TYPES,
    ORDER_TYPES,
    TRAFFIC_LEVELS,
    VEHICLE_TYPES,
    WEATHER_CONDITIONS,
)
from delivery.features import FEATURE_COLUMNS, build_training_frame
from delivery.predictor import DeliveryTimePredictor
from delivery.validation import validate_order

generate = importlib.import_module("1_generate_data")
eda = importlib.import_module("2_eda_analysis")
train = importlib.import_module("3_train_model")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    df = generate.generate_food_delivery_dataset(
        n_samples=3000, n_riders=120, n_restaurants=80, seed=7)
    path = tmp_path_factory.mktemp("data") / "food_delivery_data.csv"
    df.to_csv(path, index=False)
    return df, path


def test_generated_data_is_consistent(dataset):
    df, _ = dataset
    frame = build_training_frame(df)
    assert not frame[FEATURE_COLUMNS].isna().any().any()
    assert frame["Preparation_time_min"].between(3, 60).all()
    assert (frame["Time_taken_min"] >= frame["Preparation_time_min"] + 2).all()
    assert df["Distance_km"].between(0.3, 25.01).all()

    recomputed = generate.haversine_km(
        df["Restaurant_latitude"], df["Restaurant_longitude"],
        df["Delivery_location_latitude"], df["Delivery_location_longitude"])
    assert np.allclose(recomputed, df["Distance_km"], atol=0.01)

    assert set(df["Weather_conditions"]) <= set(WEATHER_CONDITIONS)
    assert set(df["Road_traffic_density"]) <= set(TRAFFIC_LEVELS)
    assert set(df["Type_of_vehicle"]) <= set(VEHICLE_TYPES)
    assert set(df["Type_of_order"]) <= set(ORDER_TYPES)
    assert set(df["City"]) <= set(CITY_TYPES)


def test_riders_and_restaurants_are_persistent(dataset):
    df, _ = dataset
    rider_attrs = ["Delivery_person_Age", "Delivery_person_Ratings", "Type_of_vehicle"]
    assert (df.groupby("Delivery_person_ID")[rider_attrs].nunique() == 1).all().all()
    restaurant_attrs = ["Restaurant_latitude", "Restaurant_longitude", "City"]
    assert (df.groupby("Restaurant_ID")[restaurant_attrs].nunique() == 1).all().all()


def test_generation_is_reproducible():
    first = generate.generate_food_delivery_dataset(n_samples=300, seed=1)
    second = generate.generate_food_delivery_dataset(n_samples=300, seed=1)
    pd.testing.assert_frame_equal(first, second)


def test_eda_and_training_end_to_end(dataset, tmp_path):
    _, raw_path = dataset
    viz_dir = tmp_path / "viz"
    eda.main(["--input", str(raw_path), "--output", str(tmp_path / "processed.csv"),
              "--viz-dir", str(viz_dir)])
    model_path = tmp_path / "model.joblib"
    metadata = train.main([
        "--input", str(raw_path), "--model-path", str(model_path),
        "--comparison-path", str(tmp_path / "comparison.csv"),
        "--viz-dir", str(viz_dir), "--folds", "3", "--fast",
    ])

    assert len(list(viz_dir.glob("*.png"))) == 8
    assert (tmp_path / "processed.csv").exists()
    comparison = pd.read_csv(tmp_path / "comparison.csv")
    baseline_mae = comparison.loc[comparison["Model"] == train.BASELINE_NAME, "Test MAE"].item()
    assert metadata["model_name"] != train.BASELINE_NAME
    assert metadata["test_metrics"]["mae"] < baseline_mae / 2

    predictor = DeliveryTimePredictor.load(model_path)
    result = predictor.predict(validate_order({"distance_km": 4}))
    assert result["predicted_time_min"] <= result["predicted_time"] <= result["predicted_time_max"]


def test_conformal_margin_hits_requested_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 5000)
    # Deliberately too-narrow interval: [-0.5, 0.5] covers only ~38%.
    margin = train.conformal_margin(np.full_like(y, -0.5), np.full_like(y, 0.5), y, 0.8)
    coverage = np.mean((y >= -0.5 - margin) & (y <= 0.5 + margin))
    assert 0.79 <= coverage <= 0.82
