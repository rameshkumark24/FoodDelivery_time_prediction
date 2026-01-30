from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model
try:
    model = joblib.load('models/best_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    model_info = joblib.load('models/model_info.pkl')
    print(f"✅ Model loaded: {model_info['best_model_name']}")
except:
    print("❌ Models not found. Using dummy mode.")
    model = None

# --- PHYSICS CONSTANTS ---
# Max realistic speeds in km/h for different vehicles
VEHICLE_MAX_SPEEDS = {
    'bicycle': 18,          # ~3.3 mins per km
    'electric_scooter': 30, # ~2.0 mins per km
    'scooter': 45,          # ~1.3 mins per km
    'motorcycle': 60,       # ~1.0 min per km
}

# Traffic multipliers (Higher traffic = slower speed)
# We divide max speed by this factor
TRAFFIC_FACTORS = {
    'Low': 1.0,
    'Medium': 1.2,
    'High': 1.5,
    'Jam': 2.5
}

def calculate_minimum_travel_time(distance, vehicle, traffic):
    """
    Calculates the absolute minimum time required to travel the distance
    based on physics limits of the vehicle and traffic conditions.
    """
    # Get base speed limit for vehicle (default to scooter if unknown)
    base_speed_kmh = VEHICLE_MAX_SPEEDS.get(vehicle, 45)
    
    # Apply traffic penalty
    traffic_penalty = TRAFFIC_FACTORS.get(traffic, 1.2)
    real_speed_kmh = base_speed_kmh / traffic_penalty
    
    # Calculate time: Time = Distance / Speed
    # Result in hours, convert to minutes
    travel_time_hours = distance / real_speed_kmh
    travel_time_minutes = travel_time_hours * 60
    
    return travel_time_minutes

def prepare_input_features(input_data):
    """Same feature preparation as before"""
    df = pd.DataFrame([input_data])

    if all(k in input_data for k in ['restaurant_lat', 'restaurant_lon', 'delivery_lat', 'delivery_lon']):
        df['Distance_km'] = np.sqrt(
            (input_data['restaurant_lat'] - input_data['delivery_lat'])**2 +
            (input_data['restaurant_lon'] - input_data['delivery_lon'])**2
        ) * 111

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

    age = int(input_data.get('delivery_person_age', 30))
    df['Age_group'] = 'Young' if age <= 25 else 'Middle' if age <= 35 else 'Senior'

    rating = float(input_data.get('delivery_person_ratings', 4.5))
    df['Rating_category'] = 'Average' if rating <= 4.0 else 'Good' if rating <= 4.5 else 'Excellent'

    X = pd.DataFrame(columns=feature_columns)
    
    feature_mapping = {
        'Distance_km': df['Distance_km'].values[0] if 'Distance_km' in df else float(input_data.get('distance_km', 5)),
        'Delivery_person_Age': age,
        'Delivery_person_Ratings': rating,
        'Preparation_time_min': int(input_data.get('preparation_time', 15)),
        'Order_hour': df['Order_hour'].values[0],
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
        
        # 2. Get Model Prediction (The AI Guess)
        X = prepare_input_features(input_data)
        ai_prediction = model.predict(X)[0]
        
        # 3. Calculate Physics Constraints (The Reality Check)
        min_travel_time = calculate_minimum_travel_time(distance, vehicle, traffic)
        
        # Logic: Delivery cannot happen faster than Prep Time + Travel Time
        # We add a 3-minute buffer for pickup/dropoff actions
        pickup_buffer = 3 
        min_possible_total_time = prep_time + min_travel_time + pickup_buffer
        
        # 4. Final Decision
        # If AI is too optimistic (faster than physics), use physics time
        # If AI predicts longer (due to bad weather/rating), keep AI time
        final_prediction = max(ai_prediction, min_possible_total_time)
        
        # Cap at reasonable max (e.g., 3 hours)
        final_prediction = min(180, final_prediction)

        # 5. Format Output
        uncertainty = 4 # slightly wider range for realism
        predicted_time = float(round(final_prediction, 0))
        
        return jsonify({
            'success': True,
            'predicted_time': predicted_time,
            'predicted_time_min': float(round(predicted_time - uncertainty)),
            'predicted_time_max': float(round(predicted_time + uncertainty)),
            'message': f'Estimated delivery time: {int(predicted_time)} minutes',
            'debug_info': {
                'ai_guess': round(ai_prediction, 1),
                'physics_min': round(min_possible_total_time, 1),
                'vehicle_speed_limit': VEHICLE_MAX_SPEEDS.get(vehicle, 45)
            }
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
