from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- LOAD MODELS ---
print("Loading model and encoders...")
try:
    model = joblib.load('models/best_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    model_info = joblib.load('models/model_info.pkl')
    print(f"✅ Model loaded: {model_info['best_model_name']}")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Fallback for testing without models
    model = None
    model_info = {'best_model_name': 'Physics Mode (No Model)', 'test_mae': 0}

# --- PHYSICS CONSTANTS ---
VEHICLE_MAX_SPEEDS = {
    'bicycle': 18,          # ~18 km/h
    'electric_scooter': 30, # ~30 km/h
    'scooter': 45,          # ~45 km/h
    'motorcycle': 60,       # ~60 km/h
}

TRAFFIC_FACTORS = {
    'Low': 1.0,
    'Medium': 1.2,
    'High': 1.5,
    'Jam': 2.5
}

def calculate_minimum_travel_time(distance, vehicle, traffic, age, rating):
    """
    Calculates time based on Physics + Human Factors (Age/Rating)
    """
    # 1. Base Speed (km/h)
    base_speed_kmh = VEHICLE_MAX_SPEEDS.get(vehicle, 45)
    
    # 2. Traffic Factor (Slower in traffic)
    traffic_penalty = TRAFFIC_FACTORS.get(traffic, 1.2)
    
    # 3. AGE FACTOR (Human Limit)
    # Younger people (up to 40) ride at 100% potential
    # People over 40 get 1% slower for every year
    age_factor = 1.0
    if age > 40:
        years_over = age - 40
        age_penalty = years_over * 0.01  # 1% per year
        age_factor = 1.0 - min(0.5, age_penalty) # Cap penalty at 50%
        
    # 4. RATING FACTOR (Efficiency Limit)
    # Ratings below 3.5 imply slower/less efficient service
    rating_factor = 1.0
    if rating < 3.5:
        rating_factor = 0.9 # 10% slower if rating is bad
        
    # Calculate Real Speed
    # Speed = Base * AgeFactor * RatingFactor / Traffic
    real_speed_kmh = (base_speed_kmh * age_factor * rating_factor) / traffic_penalty
    
    # Calculate Time (Minutes)
    if real_speed_kmh <= 0: real_speed_kmh = 1 # Prevent divide by zero
    travel_time_minutes = (distance / real_speed_kmh) * 60
    
    return travel_time_minutes

def prepare_input_features(input_data):
    """Prepare input data for the AI Model"""
    df = pd.DataFrame([input_data])

    # Time Features
    current_time = datetime.now()
    order_hour = int(input_data.get('order_hour', current_time.hour))
    
    df['Order_hour'] = order_hour
    df['Day_of_week'] = current_time.weekday()
    df['Month'] = current_time.month
    df['Is_weekend'] = 1 if current_time.weekday() >= 5 else 0
    df['Is_peak_hour'] = 1 if (12 <= order_hour <= 14) or (19 <= order_hour <= 21) else 0

    if 6 <= order_hour < 12: df['Time_period'] = 'Morning'
    elif 12 <= order_hour < 17: df['Time_period'] = 'Afternoon'
    elif 17 <= order_hour < 21: df['Time_period'] = 'Evening'
    else: df['Time_period'] = 'Night'

    # Age/Rating Grouping
    age = int(input_data.get('delivery_person_age', 30))
    df['Age_group'] = 'Young' if age <= 25 else 'Middle' if age <= 35 else 'Senior'

    rating = float(input_data.get('delivery_person_ratings', 4.5))
    df['Rating_category'] = 'Average' if rating <= 4.0 else 'Good' if rating <= 4.5 else 'Excellent'

    # Map to columns
    X = pd.DataFrame(columns=feature_columns)
    feature_mapping = {
        'Distance_km': float(input_data.get('distance_km', 5)),
        'Delivery_person_Age': age,
        'Delivery_person_Ratings': rating,
        'Preparation_time_min': int(input_data.get('preparation_time', 15)),
        'Order_hour': order_hour,
        'Day_of_week': df['Day_of_week'].values[0],
        'Month': df['Month'].values[0],
        'Is_weekend': df['Is_weekend'].values[0],
        'Is_peak_hour': df['Is_peak_hour'].values[0],
        'Weather_conditions': input_data.get('weather', 'Sunny'),
        'Road_traffic_density': input_data.get('traffic', 'Medium'),
        'Type_of_vehicle': input_data.get('vehicle', 'motorcycle'),
        'Type_of_order': input_data.get('order_type', 'Meal'),
        'Festival': input_data.get('festival', 'No'),
        'City': input_data.get('city', 'Urban'),
        'Time_period': df['Time_period'].values[0],
        'Age_group': df['Age_group'].values[0],
        'Rating_category': df['Rating_category'].values[0]
    }

    for feature in feature_columns:
        X[feature] = [feature_mapping.get(feature, 0)]

    # Label Encoding
    for col in X.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            try:
                X[col] = label_encoders[col].transform(X[col].astype(str))
            except:
                X[col] = 0 
    return X

@app.route('/')
def home():
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        # 1. Get Core Inputs
        prep_time = int(input_data.get('preparation_time', 15))
        distance = float(input_data.get('distance_km', 5.0))
        vehicle = input_data.get('vehicle', 'motorcycle')
        traffic = input_data.get('traffic', 'Low')
        age = int(input_data.get('delivery_person_age', 30))
        rating = float(input_data.get('delivery_person_ratings', 4.5))
        
        # 2. Get AI Prediction
        ai_prediction = 0
        if model:
            X = prepare_input_features(input_data)
            ai_prediction = model.predict(X)[0]
        
        # 3. Calculate Physics Constraints (The Logic Check)
        min_travel_time = calculate_minimum_travel_time(distance, vehicle, traffic, age, rating)
        
        # Delivery = Prep + Travel + Buffer (Pickup/Dropoff)
        pickup_buffer = 4 # Minutes
        min_possible_total_time = prep_time + min_travel_time + pickup_buffer
        
        # 4. Final Decision
        # Use the higher of the two (Physics limit vs AI prediction)
        final_prediction = max(ai_prediction, min_possible_total_time)
        
        # Hard caps for sanity
        final_prediction = min(180, final_prediction) # Max 3 hours
        
        # 5. Format Output
        uncertainty = 4 
        predicted_time = float(round(final_prediction, 0))
        
        return jsonify({
            'success': True,
            'predicted_time': predicted_time,
            'predicted_time_min': float(round(predicted_time - uncertainty)),
            'predicted_time_max': float(round(predicted_time + uncertainty)),
            'message': f'Estimated delivery time: {int(predicted_time)} minutes'
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
