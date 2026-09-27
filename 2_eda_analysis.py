"""Step 2 - exploratory data analysis and feature engineering.

Uses the same feature engineering as training and serving
(``delivery.features``), writes the processed dataset and saves the EDA plots
to ``visualizations/``.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files; no display needed
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from delivery.config import (  # noqa: E402
    DAY_NAMES,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    TARGET,
    TIME_PERIODS,
    TRAFFIC_LEVELS,
    VIZ_DIR,
)
from delivery.features import build_training_frame  # noqa: E402

DPI = 120


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Columns that are handy for plots but are not model features."""
    df = df.copy()
    df["Day_name"] = df["Day_of_week"].map(dict(enumerate(DAY_NAMES)))
    df["Distance_category"] = pd.cut(df["Distance_km"], bins=[0, 3, 6, 10, float("inf")],
                                     labels=["Very close (<3 km)", "Close (3-6 km)",
                                             "Moderate (6-10 km)", "Far (>10 km)"])
    df["Age_group"] = pd.cut(df["Delivery_person_Age"], bins=[0, 25, 35, 100],
                             labels=["Young", "Middle", "Senior"])
    df["Rating_category"] = pd.cut(df["Delivery_person_Ratings"], bins=[0, 4.0, 4.5, 5.0],
                                   labels=["Average", "Good", "Excellent"])
    return df


def save(fig, out_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def plot_target_distribution(df, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist(df[TARGET], bins=60, color="skyblue", edgecolor="black")
    axes[0].axvline(df[TARGET].mean(), color="red", linestyle="--",
                    label=f"Mean: {df[TARGET].mean():.1f} min")
    axes[0].axvline(df[TARGET].median(), color="green", linestyle=":",
                    label=f"Median: {df[TARGET].median():.1f} min")
    axes[0].set(xlabel="Delivery time (minutes)", ylabel="Orders",
                title="Distribution of delivery time")
    axes[0].legend()
    axes[1].boxplot(df[TARGET])
    axes[1].set(ylabel="Delivery time (minutes)", title="Delivery time box plot")
    save(fig, out_dir, "01_target_distribution.png")


def plot_categorical_impact(df, out_dir):
    features = ["Weather_conditions", "Road_traffic_density", "Type_of_vehicle",
                "Type_of_order", "Time_period", "City"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    for ax, feature in zip(axes.flat, features):
        df.groupby(feature)[TARGET].mean().sort_values().plot(kind="barh", ax=ax, color="coral")
        ax.set(xlabel="Average delivery time (min)", ylabel="", title=f"Impact of {feature}")
        ax.grid(axis="x", alpha=0.3)
    save(fig, out_dir, "02_categorical_impact.png")


def plot_correlations(df, out_dir):
    numeric = ["Distance_km", "Preparation_time_min", "Delivery_person_Age",
               "Delivery_person_Ratings", "Order_hour", "Is_peak_hour", TARGET]
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(df[numeric].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=1, ax=ax)
    ax.set_title("Feature correlation matrix")
    save(fig, out_dir, "03_correlation_matrix.png")


def plot_time_analysis(df, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    df.groupby("Order_hour")[TARGET].mean().plot(ax=axes[0], marker="o", color="green")
    axes[0].set(xlabel="Hour of day", ylabel="Average delivery time (min)",
                title="Delivery time by hour", xticks=range(0, 24, 2))
    df.groupby("Day_name")[TARGET].mean().reindex(DAY_NAMES).plot(ax=axes[1], marker="o",
                                                                  color="purple")
    axes[1].set(xlabel="", ylabel="Average delivery time (min)", title="Delivery time by day")
    axes[1].tick_params(axis="x", rotation=45)
    peak = df.groupby("Is_peak_hour")[TARGET].mean()
    axes[2].bar(["Non-peak", "Peak"], peak.reindex([0, 1]).values, color=["lightblue", "salmon"])
    axes[2].set(ylabel="Average delivery time (min)", title="Peak hour impact")
    for ax in axes:
        ax.grid(alpha=0.3)
    save(fig, out_dir, "04_time_analysis.png")


def plot_distance_vs_time(df, out_dir):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=df.sample(min(len(df), 5000), random_state=0), x="Distance_km",
                    y=TARGET, hue="Road_traffic_density", hue_order=TRAFFIC_LEVELS,
                    alpha=0.45, s=18, ax=ax)
    ax.set(xlabel="Distance (km)", ylabel="Delivery time (minutes)",
           title="Distance vs delivery time (coloured by traffic)")
    ax.grid(alpha=0.3)
    save(fig, out_dir, "05_distance_vs_time.png")


def print_summary(df):
    print("\nDelivery time statistics (minutes):")
    print(df[TARGET].describe().round(2).to_string())
    for feature in ["Road_traffic_density", "Weather_conditions", "Type_of_vehicle"]:
        print(f"\nAverage delivery time by {feature}:")
        print(df.groupby(feature)[TARGET].mean().sort_values(ascending=False).round(1).to_string())
    by_period = df.groupby("Time_period")[TARGET].mean().reindex(TIME_PERIODS).round(1)
    print("\nAverage delivery time by time of day:")
    print(by_period.to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(RAW_DATA_PATH))
    parser.add_argument("--output", default=str(PROCESSED_DATA_PATH))
    parser.add_argument("--viz-dir", default=str(VIZ_DIR))
    args = parser.parse_args(argv)

    sns.set_style("whitegrid")
    raw = pd.read_csv(args.input)
    print(f"Loaded {len(raw):,} rows x {raw.shape[1]} columns from {args.input}")

    missing = raw.isna().sum()
    print("\nMissing values:", "none" if missing.sum() == 0 else "")
    if missing.sum():
        print(missing[missing > 0].to_string())

    df = add_analysis_columns(build_training_frame(raw))
    print(f"\nFeature engineering added {df.shape[1] - raw.shape[1]} columns:")
    for column in sorted(set(df.columns) - set(raw.columns)):
        print(f"  - {column}")

    out_dir = Path(args.viz_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCreating visualizations in {out_dir}/")
    plot_target_distribution(df, out_dir)
    plot_categorical_impact(df, out_dir)
    plot_correlations(df, out_dir)
    plot_time_analysis(df, out_dir)
    plot_distance_vs_time(df, out_dir)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n✅ Processed data saved to {args.output}")
    print_summary(df)


if __name__ == "__main__":
    main()
