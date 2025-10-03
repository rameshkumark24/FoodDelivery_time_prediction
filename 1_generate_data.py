import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_food_delivery_dataset(n_samples=10000):
    """
    Generate realistic food delivery dataset with multiple features
    """
    
    # Base date
    start_date = datetime(2024, 1, 1)
    
    data = {
        'Order_ID': [f'ORD{str(i).zfill(6)}' for i in range(1, n_samples + 1)],
        'Delivery_person_ID': [f'DEL{random.randint(1, 100):03d}' for _ in range(n_samples)],
        'Delivery_person_Age': np.random.randint(20, 45, n_samples),
        'Delivery_person_Ratings': np.round(np.random.uniform(3.5, 5.0, n_samples), 1),
        'Restaurant_latitude': np.random.uniform(12.9000, 13.1000, n_samples),
        'Restaurant_longitude': np.random.uniform(77.5000, 77.7000, n_samples),
        'Delivery_location_latitude': np.random.uniform(12.9000, 13.1000, n_samples),
        'Delivery_location_longitude': np.random.uniform(77.5000, 77.7000, n_samples),
        'Type_of_order': np.random.choice(['Snack', 'Meal', 'Drinks', 'Buffet'], n_samples, p=[0.3, 0.5, 0.15, 0.05]),
        'Type_of_vehicle': np.random.choice(['motorcycle', 'scooter', 'bicycle', 'electric_scooter'], n_samples, p=[0.5, 0.3, 0.15, 0.05]),
        'Weather_conditions': np.random.choice(['Sunny', 'Cloudy', 'Rainy', 'Fog', 'Stormy'], n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05]),
        'Road_traffic_density': np.random.choice(['Low', 'Medium', 'High', 'Jam'], n_samples, p=[0.25, 0.35, 0.3, 0.1]),
        'Festival': np.random.choice(['No', 'Yes'], n_samples, p=[0.85, 0.15]),
        'City': np.random.choice(['Urban', 'Semi-Urban', 'Metropolitan'], n_samples, p=[0.4, 0.3, 0.3]),
    }
    
    # Generate order dates and times
    order_dates = []
    order_times = []
    
    for _ in range(n_samples):
        date = start_date + timedelta(days=random.randint(0, 365))
        # Peak hours: 12-14 (lunch) and 19-21 (dinner)
        if random.random() < 0.6:  # 60% orders during peak
            hour = random.choice(list(range(12, 15)) + list(range(19, 22)))
        else:
            hour = random.randint(8, 23)
        
        minute = random.randint(0, 59)
        time = f"{hour:02d}:{minute:02d}"
        
        order_dates.append(date.strftime('%Y-%m-%d'))
        order_times.append(time)
    
    data['Order_Date'] = order_dates
    data['Time_Orderd'] = order_times
    data['Time_Order_picked'] = [
        (datetime.strptime(t, '%H:%M') + timedelta(minutes=random.randint(10, 30))).strftime('%H:%M')
        for t in order_times
    ]
    
    df = pd.DataFrame(data)
    
    # Calculate distance (in km) using Haversine formula approximation
    df['Distance_km'] = np.sqrt(
        (df['Restaurant_latitude'] - df['Delivery_location_latitude'])**2 +
        (df['Restaurant_longitude'] - df['Delivery_location_longitude'])**2
    ) * 111  # Rough conversion to km
    
    # Generate realistic delivery time (in minutes)
    base_time = df['Distance_km'] * 3  # 3 min per km base
    
    # Add factors
    traffic_multiplier = df['Road_traffic_density'].map({
        'Low': 1.0, 'Medium': 1.3, 'High': 1.6, 'Jam': 2.0
    })
    
    weather_multiplier = df['Weather_conditions'].map({
        'Sunny': 1.0, 'Cloudy': 1.1, 'Rainy': 1.4, 'Fog': 1.3, 'Stormy': 1.6
    })
    
    vehicle_multiplier = df['Type_of_vehicle'].map({
        'motorcycle': 1.0, 'scooter': 1.1, 'bicycle': 1.5, 'electric_scooter': 1.2
    })
    
    order_complexity = df['Type_of_order'].map({
        'Snack': 1.0, 'Meal': 1.2, 'Drinks': 0.9, 'Buffet': 1.5
    })
    
    festival_multiplier = df['Festival'].map({'No': 1.0, 'Yes': 1.3})
    
    # Rating effect (better ratings = faster delivery)
    rating_effect = 2 - (df['Delivery_person_Ratings'] / 5)
    
    # Calculate final time
    delivery_time = (
        base_time * 
        traffic_multiplier * 
        weather_multiplier * 
        vehicle_multiplier * 
        order_complexity * 
        festival_multiplier * 
        rating_effect
    )
    
    # Add some random noise
    delivery_time = delivery_time + np.random.normal(0, 3, n_samples)
    
    # Ensure minimum time and convert to minutes
    df['Time_taken_min'] = np.clip(delivery_time, 15, 90).round(0).astype(int)
    
    return df

# Generate dataset
print("Generating realistic food delivery dataset...")
df = generate_food_delivery_dataset(10000)

# Save to CSV
df.to_csv('food_delivery_data.csv', index=False)

print(f"\n✅ Dataset generated successfully!")
print(f"Total samples: {len(df)}")
print(f"\nDataset preview:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nStatistics:")
print(df.describe())
print(f"\nTarget variable distribution (Time_taken_min):")
print(df['Time_taken_min'].describe())

# Save a sample for quick testing
df.sample(1000).to_csv('food_delivery_sample.csv', index=False)
print(f"\n✅ Sample dataset (1000 rows) saved as 'food_delivery_sample.csv'")