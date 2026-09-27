"""Step 3 - train, compare and save delivery time models.

* Candidates are compared with K-fold cross-validation on the training split
  only; the held-out test split is used once, to report final performance.
* Each model is a scikit-learn ``Pipeline`` (categorical encoding + regressor),
  so the encoding can never drift out of sync with the model.
* Two quantile models provide an 80% prediction interval per order, widened
  by split-conformal calibration (CQR) so the coverage actually holds.
* Everything the app needs is saved as a single bundle:
  ``models/delivery_model.joblib``.
"""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lightgbm import LGBMRegressor  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import KFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OrdinalEncoder  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

from delivery.config import (  # noqa: E402
    CITY_TYPES,
    COMPARISON_PATH,
    FESTIVAL_OPTIONS,
    INTERVAL_LOWER_QUANTILE,
    INTERVAL_UPPER_QUANTILE,
    MODEL_PATH,
    ORDER_TYPES,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET,
    TIME_PERIODS,
    TRAFFIC_LEVELS,
    VEHICLE_TYPES,
    VIZ_DIR,
    WEATHER_CONDITIONS,
)
from delivery.features import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    build_training_frame,
)
from delivery.predictor import (  # noqa: E402
    BUNDLE_FORMAT_VERSION,
    installed_versions,
    minimum_feasible_minutes,
)

BASELINE_NAME = "Baseline (median)"
CATEGORY_LEVELS = {
    "Weather_conditions": WEATHER_CONDITIONS,
    "Road_traffic_density": TRAFFIC_LEVELS,
    "Type_of_vehicle": VEHICLE_TYPES,
    "Type_of_order": ORDER_TYPES,
    "Festival": FESTIVAL_OPTIONS,
    "City": CITY_TYPES,
    "Time_period": TIME_PERIODS,
}
DPI = 120


def banner(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def make_preprocessor():
    encoder = OrdinalEncoder(
        categories=[CATEGORY_LEVELS[c] for c in CATEGORICAL_FEATURES],
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )
    return ColumnTransformer(
        [("categorical", encoder, CATEGORICAL_FEATURES),
         ("numeric", "passthrough", NUMERIC_FEATURES)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_pipeline(regressor):
    return Pipeline([("preprocess", make_preprocessor()), ("model", regressor)])


def candidate_models(fast=False):
    n_trees = 60 if fast else None  # small ensembles for smoke tests
    return {
        BASELINE_NAME: DummyRegressor(strategy="median"),
        "Random Forest": RandomForestRegressor(
            n_estimators=n_trees or 200, min_samples_leaf=5, max_features=0.6,
            n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=n_trees or 400, max_depth=4, learning_rate=0.05, subsample=0.8,
            random_state=RANDOM_STATE),
        "XGBoost": XGBRegressor(
            n_estimators=n_trees or 1000, max_depth=4, learning_rate=0.03, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3, n_jobs=-1, random_state=RANDOM_STATE),
        "LightGBM": LGBMRegressor(
            n_estimators=n_trees or 1200, num_leaves=15, learning_rate=0.03, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, min_child_samples=20,
            importance_type="gain", n_jobs=-1, random_state=RANDOM_STATE, verbose=-1),
    }


def quantile_model(quantile, fast=False):
    return make_pipeline(HistGradientBoostingRegressor(
        loss="quantile", quantile=quantile, max_iter=100 if fast else 400,
        learning_rate=0.05, random_state=RANDOM_STATE))


def conformal_margin(lower_pred, upper_pred, y_true, confidence):
    """Split-conformal (CQR) adjustment: how far both interval ends must move
    so that ``confidence`` of calibration orders fall inside the interval."""
    y_true = np.asarray(y_true)
    scores = np.maximum(lower_pred - y_true, y_true - upper_pred)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * confidence) / n)
    return float(np.quantile(scores, level, method="higher"))


def guardrail_floor(X):
    return minimum_feasible_minutes(
        X["Distance_km"].to_numpy(), X["Preparation_time_min"].to_numpy(),
        X["Type_of_vehicle"].to_numpy())


def regression_metrics(y_true, y_pred):
    errors = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "within_5_min_pct": float(np.mean(errors <= 5) * 100),
        "within_10_min_pct": float(np.mean(errors <= 10) * 100),
    }


def cross_validate_candidates(candidates, X_train, y_train, folds):
    cv = KFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error",
               "r2": "r2"}
    cv_results = {}
    for name, regressor in candidates.items():
        start = time.perf_counter()
        scores = cross_validate(make_pipeline(regressor), X_train, y_train, cv=cv,
                                scoring=scoring)
        cv_results[name] = {
            "cv_mae": float(-scores["test_mae"].mean()),
            "cv_mae_std": float(scores["test_mae"].std()),
            "cv_rmse": float(-scores["test_rmse"].mean()),
            "cv_r2": float(scores["test_r2"].mean()),
        }
        r = cv_results[name]
        print(f"  {name:<20} CV MAE {r['cv_mae']:6.2f} ± {r['cv_mae_std']:.2f} min | "
              f"RMSE {r['cv_rmse']:6.2f} | R² {r['cv_r2']:.4f} "
              f"({time.perf_counter() - start:.1f}s)")
    return cv_results


def feature_importances(pipeline):
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return {}
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    values = np.asarray(model.feature_importances_, dtype=float)
    values = values / values.sum() if values.sum() else values
    order = np.argsort(values)[::-1]
    return {str(names[i]): round(float(values[i]), 4) for i in order}


def plot_feature_importance(importances, model_name, out_dir):
    if not importances:
        return
    top = pd.Series(importances).head(15)[::-1]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.values, color="steelblue")
    ax.set(xlabel="Relative importance", title=f"Feature importance - {model_name}")
    fig.tight_layout()
    fig.savefig(out_dir / "06_feature_importance.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_analysis(y_test, y_pred, lower, upper, model_name, out_dir):
    y_test = np.asarray(y_test)
    residuals = y_test - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.35, s=12)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, "r--", lw=2)
    axes[0].set(xlabel="Actual delivery time (min)", ylabel="Predicted (min)",
                title=f"Actual vs predicted - {model_name}")

    axes[1].hist(residuals, bins=60, color="lightcoral", edgecolor="black")
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].set(xlabel="Actual - predicted (min)", ylabel="Orders",
                title="Distribution of prediction errors")

    order = np.argsort(y_pred)
    sample = order[:: max(1, len(order) // 300)]
    axes[2].fill_between(range(len(sample)), lower[sample], upper[sample], color="orange",
                         alpha=0.3, label="80% interval")
    axes[2].plot(range(len(sample)), y_pred[sample], color="darkorange", label="Prediction")
    axes[2].scatter(range(len(sample)), y_test[sample], s=8, color="black", label="Actual")
    axes[2].set(xlabel="Test orders (sorted by prediction)", ylabel="Minutes",
                title="Prediction intervals")
    axes[2].legend()

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "07_prediction_analysis.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison, out_dir):
    df = comparison.sort_values("CV MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["Model"], df["CV MAE"], xerr=df["CV MAE std"], color="mediumseagreen",
            capsize=4)
    ax.set(xlabel="Cross-validated MAE (minutes, lower is better)", title="Model comparison")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "08_model_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(RAW_DATA_PATH))
    parser.add_argument("--model-path", default=str(MODEL_PATH))
    parser.add_argument("--comparison-path", default=str(COMPARISON_PATH))
    parser.add_argument("--viz-dir", default=str(VIZ_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fast", action="store_true",
                        help="small ensembles for quick smoke tests")
    args = parser.parse_args(argv)

    banner("FOOD DELIVERY TIME PREDICTION - MODEL TRAINING")
    raw = pd.read_csv(args.input)
    df = build_training_frame(raw)
    X, y = df[FEATURE_COLUMNS], df[TARGET]
    print(f"Loaded {len(df):,} orders from {args.input}")
    print(f"{len(FEATURE_COLUMNS)} features: {', '.join(FEATURE_COLUMNS)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train):,} orders | Test (held out): {len(X_test):,} orders")

    banner(f"{args.folds}-FOLD CROSS-VALIDATION (training split only)")
    candidates = candidate_models(args.fast)
    cv_results = cross_validate_candidates(candidates, X_train, y_train, args.folds)
    best_name = min((n for n in cv_results if n != BASELINE_NAME),
                    key=lambda n: cv_results[n]["cv_mae"])
    print(f"\n🏆 Selected by CV MAE: {best_name}")

    banner("HELD-OUT TEST PERFORMANCE")
    rows, best_pipeline = [], None
    for name, regressor in candidates.items():
        pipeline = make_pipeline(regressor).fit(X_train, y_train)
        test = regression_metrics(y_test, pipeline.predict(X_test))
        rows.append({
            "Model": name,
            "CV MAE": cv_results[name]["cv_mae"],
            "CV MAE std": cv_results[name]["cv_mae_std"],
            "Test MAE": test["mae"],
            "Test RMSE": test["rmse"],
            "Test R²": test["r2"],
            "Within ±5 min (%)": test["within_5_min_pct"],
            "Within ±10 min (%)": test["within_10_min_pct"],
        })
        if name == best_name:
            best_pipeline = pipeline
    comparison = pd.DataFrame(rows).sort_values("CV MAE").reset_index(drop=True)
    print(comparison.round(3).to_string(index=False))

    # Evaluate exactly what the app serves: model output raised to the guardrail.
    floor = guardrail_floor(X_test)
    raw_pred = best_pipeline.predict(X_test)
    y_pred = np.maximum(raw_pred, floor)
    test_metrics = regression_metrics(y_test, y_pred)
    guardrail_rate = float(np.mean(raw_pred < floor) * 100)

    confidence = round(INTERVAL_UPPER_QUANTILE - INTERVAL_LOWER_QUANTILE, 2)
    banner(f"{confidence:.0%} PREDICTION INTERVALS (conformalized quantile regression)")
    # Quantile models are fit on part of the training split; the rest
    # calibrates them so the interval's coverage is honest.
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE)
    lower_model = quantile_model(INTERVAL_LOWER_QUANTILE, args.fast).fit(X_fit, y_fit)
    upper_model = quantile_model(INTERVAL_UPPER_QUANTILE, args.fast).fit(X_fit, y_fit)
    margin = conformal_margin(lower_model.predict(X_cal), upper_model.predict(X_cal),
                              y_cal, confidence)

    lower_pred, upper_pred = lower_model.predict(X_test), upper_model.predict(X_test)

    def interval(adjustment):
        # Same clamping as DeliveryTimePredictor.predict.
        lower = np.minimum(np.maximum(lower_pred - adjustment, floor), y_pred)
        upper = np.maximum(upper_pred + adjustment, y_pred)
        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)) * 100)
        return lower, upper, coverage

    _, _, raw_coverage = interval(0.0)
    lower, upper, coverage = interval(margin)
    mean_width = float(np.mean(upper - lower))
    print(f"  Uncalibrated test coverage: {raw_coverage:.1f}%")
    print(f"  Conformal margin:           {margin:+.2f} min per side")
    print(f"  Calibrated test coverage:   {coverage:.1f}% (target {confidence:.0%})")
    print(f"  Mean interval width:        {mean_width:.1f} min")

    residuals = np.asarray(y_test) - y_pred
    banner(f"FINAL MODEL: {best_name}")
    print(f"  Test MAE:            {test_metrics['mae']:.2f} min")
    print(f"  Test RMSE:           {test_metrics['rmse']:.2f} min")
    print(f"  Test R²:             {test_metrics['r2']:.4f}")
    print(f"  Within ±5 min:       {test_metrics['within_5_min_pct']:.1f}%")
    print(f"  Within ±10 min:      {test_metrics['within_10_min_pct']:.1f}%")
    print(f"  Mean error (bias):   {residuals.mean():+.2f} min")
    print(f"  Guardrail triggered: {guardrail_rate:.2f}% of test orders")

    importances = feature_importances(best_pipeline)
    if importances:
        print("\nTop features:")
        for feature, value in list(importances.items())[:8]:
            print(f"  {feature:<26} {value:.3f}")

    viz_dir = Path(args.viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(importances, best_name, viz_dir)
    plot_prediction_analysis(y_test, y_pred, lower, upper, best_name, viz_dir)
    plot_model_comparison(comparison, viz_dir)
    print(f"\n✅ Plots saved to {viz_dir}/")

    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "cv_folds": args.folds,
        "cv": cv_results[best_name],
        "test_metrics": test_metrics,
        "guardrail_triggered_pct": guardrail_rate,
        "interval": {
            "confidence": confidence,
            "conformal_margin_min": margin,
            "uncalibrated_test_coverage_pct": raw_coverage,
            "test_coverage_pct": coverage,
            "mean_width_min": mean_width,
        },
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": importances,
        "comparison": comparison.to_dict(orient="records"),
        "versions": installed_versions(),
    }
    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION,
        "model": best_pipeline,
        "lower_model": lower_model,
        "upper_model": upper_model,
        "interval_margin": margin,
        "metadata": metadata,
    }
    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    Path(args.comparison_path).parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.comparison_path, index=False)
    print(f"✅ Model bundle saved to {model_path} "
          f"({model_path.stat().st_size / 1e6:.1f} MB)")
    print(f"✅ Model comparison saved to {args.comparison_path}")
    return metadata


if __name__ == "__main__":
    main()
