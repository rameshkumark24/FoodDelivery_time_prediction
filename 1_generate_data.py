"""Step 1 - generate a realistic synthetic food delivery dataset.

The simulated delivery time is the sum of its real-world parts:

    Time_taken_min = kitchen preparation + pickup handover
                     + travel (distance / vehicle speed, slowed by traffic,
                       weather, city density, festivals, rider skill...)
                     + drop-off handover + noise (+ rare incidents)

so every feature the web app asks for genuinely influences the target.

Riders and restaurants are persistent entities: a rider keeps the same age,
rating and vehicle across all their orders, and a restaurant has a fixed
location, city type and kitchen speed.
"""
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from delivery.config import (
    CITY_TYPES,
    ORDER_TYPES,
    PEAK_HOURS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SAMPLE_DATA_PATH,
    TRAFFIC_LEVELS,
    VEHICLE_TYPES,
    WEATHER_CONDITIONS,
)

EARTH_RADIUS_KM = 6371.0
CITY_CENTER = (12.9716, 77.5946)  # Bengaluru
START_DATE = datetime(2024, 1, 1)

# Relative order volume for each hour of the day (lunch and dinner peaks).
HOURLY_ORDER_WEIGHTS = np.array([
    0.8, 0.5, 0.3, 0.2, 0.2, 0.3,   # 00-05
    0.6, 1.2, 2.0, 2.5, 2.5, 3.5,   # 06-11
    7.0, 7.5, 6.0, 3.5, 3.0, 3.5,   # 12-17
    5.0, 8.0, 9.0, 7.5, 4.0, 2.0,   # 18-23
])

# Effective average speed on city roads (includes the road/straight-line factor).
VEHICLE_SPEED_KMH = {"motorcycle": 24, "scooter": 21, "electric_scooter": 18, "bicycle": 12}
TRAFFIC_TIME_FACTOR = {"Low": 1.0, "Medium": 1.2, "High": 1.5, "Jam": 2.0}
WEATHER_TIME_FACTOR = {"Sunny": 1.0, "Cloudy": 1.05, "Rainy": 1.3, "Fog": 1.2, "Stormy": 1.55}
CITY_TIME_FACTOR = {"Semi-Urban": 0.9, "Urban": 1.0, "Metropolitan": 1.12}
DROPOFF_BASE_MIN = {"Semi-Urban": 2.0, "Urban": 3.0, "Metropolitan": 4.0}  # high-rises take longer
PREP_BASE_MIN = {"Snack": 10, "Meal": 17, "Drinks": 6, "Buffet": 26}
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]
TRAFFIC_PROBS = {  # P(Low, Medium, High, Jam)
    "night": [0.6, 0.3, 0.08, 0.02],
    "peak": [0.1, 0.3, 0.4, 0.2],
    "regular": [0.3, 0.4, 0.22, 0.08],
}
METRO_TRAFFIC_SHIFT = np.array([-0.05, -0.05, 0.05, 0.05])


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def destination_point(lat, lon, bearing_rad, distance_km):
    """Great-circle destination from a start point, bearing and distance."""
    lat1, lon1 = np.radians(lat), np.radians(lon)
    d = distance_km / EARTH_RADIUS_KM
    lat2 = np.arcsin(np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(bearing_rad))
    lon2 = lon1 + np.arctan2(np.sin(bearing_rad) * np.sin(d) * np.cos(lat1),
                             np.cos(d) - np.sin(lat1) * np.sin(lat2))
    return np.degrees(lat2), np.degrees(lon2)


def make_riders(rng, n_riders):
    skill = rng.normal(0, 1, n_riders)
    # Mostly riders in their 20s-30s, with a uniform tail so every age is covered.
    age = np.where(rng.random(n_riders) < 0.75,
                   rng.normal(28, 5, n_riders), rng.uniform(18, 60, n_riders))
    return pd.DataFrame({
        "Delivery_person_ID": [f"DEL{i:04d}" for i in range(1, n_riders + 1)],
        "Delivery_person_Age": np.clip(np.round(age), 18, 60).astype(int),
        # Ratings reflect skill, but imperfectly.
        "Delivery_person_Ratings": np.clip(
            np.round(4.2 + 0.4 * skill + rng.normal(0, 0.3, n_riders), 1), 2.0, 5.0),
        "Type_of_vehicle": rng.choice(VEHICLE_TYPES, n_riders, p=[0.5, 0.3, 0.12, 0.08]),
        "skill": skill,
    })


def make_restaurants(rng, n_restaurants):
    return pd.DataFrame({
        "Restaurant_ID": [f"RES{i:04d}" for i in range(1, n_restaurants + 1)],
        "Restaurant_latitude": CITY_CENTER[0] + rng.uniform(-0.12, 0.12, n_restaurants),
        "Restaurant_longitude": CITY_CENTER[1] + rng.uniform(-0.12, 0.12, n_restaurants),
        "City": rng.choice(CITY_TYPES, n_restaurants, p=[0.4, 0.3, 0.3]),
        "kitchen_speed": rng.lognormal(0, 0.2, n_restaurants),
    })


def sample_traffic(rng, hours, cities):
    period = np.where(np.isin(hours, NIGHT_HOURS), "night",
                      np.where(np.isin(hours, list(PEAK_HOURS)), "peak", "regular"))
    metro = cities == "Metropolitan"
    traffic = np.empty(len(hours), dtype=object)
    for name, probs in TRAFFIC_PROBS.items():
        for is_metro in (False, True):
            mask = (period == name) & (metro == is_metro)
            # Metropolitan roads are more congested: shift mass from Low/Medium to High/Jam.
            p = np.clip(np.array(probs) + (METRO_TRAFFIC_SHIFT if is_metro else 0), 0, None)
            traffic[mask] = rng.choice(TRAFFIC_LEVELS, mask.sum(), p=p / p.sum())
    return traffic


def sample_weather(rng, months):
    dry = [0.45, 0.30, 0.10, 0.10, 0.05]
    monsoon = [0.20, 0.30, 0.30, 0.05, 0.15]  # June - September
    is_monsoon = np.isin(months, [6, 7, 8, 9])
    return np.where(is_monsoon,
                    rng.choice(WEATHER_CONDITIONS, len(months), p=monsoon),
                    rng.choice(WEATHER_CONDITIONS, len(months), p=dry))


def generate_food_delivery_dataset(n_samples=20000, n_riders=600, n_restaurants=300,
                                   seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    riders = make_riders(rng, n_riders)
    restaurants = make_restaurants(rng, n_restaurants)

    df = pd.concat([
        restaurants.iloc[rng.integers(0, n_restaurants, n_samples)].reset_index(drop=True),
        riders.iloc[rng.integers(0, n_riders, n_samples)].reset_index(drop=True),
    ], axis=1)
    df.insert(0, "Order_ID", [f"ORD{i:06d}" for i in range(1, n_samples + 1)])

    # --- When -------------------------------------------------------------
    day_offsets = rng.integers(0, 366, n_samples)
    festival_days = set(rng.choice(366, 30, replace=False).tolist())
    hours = rng.choice(24, n_samples, p=HOURLY_ORDER_WEIGHTS / HOURLY_ORDER_WEIGHTS.sum())
    minutes = rng.integers(0, 60, n_samples)
    ordered_at = [START_DATE + timedelta(days=int(d), hours=int(h), minutes=int(m))
                  for d, h, m in zip(day_offsets, hours, minutes)]
    months = np.array([t.month for t in ordered_at])
    weekdays = np.array([t.weekday() for t in ordered_at])
    peak = np.isin(hours, list(PEAK_HOURS))
    night = np.isin(hours, NIGHT_HOURS)

    df["Festival"] = np.where(np.isin(day_offsets, list(festival_days)), "Yes", "No")
    festival = df["Festival"].eq("Yes").to_numpy()
    df["Type_of_order"] = rng.choice(ORDER_TYPES, n_samples, p=[0.3, 0.5, 0.15, 0.05])
    df["Weather_conditions"] = sample_weather(rng, months)
    df["Road_traffic_density"] = sample_traffic(rng, hours, df["City"].to_numpy())

    # --- Where ------------------------------------------------------------
    # Mostly short hops, plus a uniform tail so long trips are covered too.
    distance = np.where(rng.random(n_samples) < 0.9,
                        rng.lognormal(np.log(4.5), 0.55, n_samples),
                        rng.uniform(0.5, 25, n_samples))
    distance = np.clip(distance, 0.3, 25)
    bearing = rng.uniform(0, 2 * np.pi, n_samples)
    dest_lat, dest_lon = destination_point(df["Restaurant_latitude"].to_numpy(),
                                           df["Restaurant_longitude"].to_numpy(), bearing, distance)
    df["Delivery_location_latitude"] = dest_lat
    df["Delivery_location_longitude"] = dest_lon
    df["Distance_km"] = haversine_km(df["Restaurant_latitude"], df["Restaurant_longitude"],
                                     dest_lat, dest_lon)

    # --- Kitchen ----------------------------------------------------------
    prep = (df["Type_of_order"].map(PREP_BASE_MIN).to_numpy()
            * df["kitchen_speed"].to_numpy()
            * np.where(peak, 1.15, 1.0)
            * np.where(festival, 1.2, 1.0)
            + rng.normal(0, 2.5, n_samples))
    backlog = rng.random(n_samples) < 0.03  # large orders / kitchen backlog
    prep = np.where(backlog, rng.uniform(20, 60, n_samples), prep)
    prep = np.clip(np.round(prep), 3, 60).astype(int)
    picked_at = [t + timedelta(minutes=int(p)) for t, p in zip(ordered_at, prep)]

    df["Order_Date"] = [t.strftime("%Y-%m-%d") for t in ordered_at]
    df["Time_Orderd"] = [t.strftime("%H:%M") for t in ordered_at]
    df["Time_Order_picked"] = [t.strftime("%H:%M") for t in picked_at]

    # --- Road -------------------------------------------------------------
    vehicle = df["Type_of_vehicle"].to_numpy()
    weather = df["Weather_conditions"].to_numpy()
    speed = df["Type_of_vehicle"].map(VEHICLE_SPEED_KMH).to_numpy()
    travel = (distance / speed * 60
              * df["Road_traffic_density"].map(TRAFFIC_TIME_FACTOR).to_numpy()
              * df["Weather_conditions"].map(WEATHER_TIME_FACTOR).to_numpy()
              * df["City"].map(CITY_TIME_FACTOR).to_numpy()
              * np.where(festival, 1.15, 1.0)
              * np.where(night, 0.9, 1.0)
              * np.where(weekdays >= 5, 0.97, 1.0)
              * np.where((vehicle == "bicycle") & np.isin(weather, ["Rainy", "Stormy"]), 1.1, 1.0)
              * np.exp(-0.06 * df["skill"].to_numpy())
              * (1 + 0.004 * np.clip(df["Delivery_person_Age"].to_numpy() - 35, 0, None)))

    pickup_handover = rng.uniform(1, 4, n_samples)
    dropoff = (df["City"].map(DROPOFF_BASE_MIN).to_numpy()
               + np.where(df["Type_of_order"].eq("Buffet"), 2.0, 0.0)
               + rng.exponential(1.0, n_samples))
    noise = rng.normal(0, 1.0 + 0.08 * travel)
    incident = np.where(rng.random(n_samples) < 0.02, rng.uniform(8, 25, n_samples), 0.0)

    total = prep + pickup_handover + travel + dropoff + noise + incident
    df["Time_taken_min"] = np.clip(np.round(total), prep + 2, 240).astype(int)

    columns = [
        "Order_ID", "Delivery_person_ID", "Delivery_person_Age", "Delivery_person_Ratings",
        "Restaurant_ID", "Restaurant_latitude", "Restaurant_longitude",
        "Delivery_location_latitude", "Delivery_location_longitude",
        "Type_of_order", "Type_of_vehicle", "Weather_conditions", "Road_traffic_density",
        "Festival", "City", "Order_Date", "Time_Orderd", "Time_Order_picked",
        "Distance_km", "Time_taken_min",
    ]
    df = df[columns]
    return df.round({
        "Restaurant_latitude": 6, "Restaurant_longitude": 6,
        "Delivery_location_latitude": 6, "Delivery_location_longitude": 6,
        "Distance_km": 2,
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--output", default=str(RAW_DATA_PATH))
    parser.add_argument("--sample-output", default=str(SAMPLE_DATA_PATH))
    args = parser.parse_args(argv)

    print(f"Generating {args.samples:,} synthetic food delivery orders...")
    df = generate_food_delivery_dataset(args.samples, seed=args.seed)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    df.sample(min(1000, len(df)), random_state=args.seed).to_csv(args.sample_output, index=False)

    print(f"\n✅ Saved {len(df):,} rows to {args.output}")
    print(f"✅ Saved a 1,000-row sample to {args.sample_output}")
    print("\nPreview:")
    print(df.head().to_string())
    print("\nDelivery time (minutes):")
    print(df["Time_taken_min"].describe().round(2).to_string())
    print("\nDistance (km):")
    print(df["Distance_km"].describe().round(2).to_string())


if __name__ == "__main__":
    main()
