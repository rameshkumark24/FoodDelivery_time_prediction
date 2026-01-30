from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Load model and encoders
print("Loading model and encoders...")
try:
    model = joblib.load('models/best_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    feature_columns = joblib.load('models/feature_columns.pkl')
    model_info = joblib.load('models/model_info.pkl')
    print(f"✅ Model loaded: {model_info['best_model_name']}")
    print(f"✅ Test MAE: {model_info['test_mae']:.2f} minutes")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    # Fallback for testing without models
    model_info = {'best_model_name': 'Not Loaded', 'test_mae': 0}

def prepare_input_features(input_data):
    """Prepare input data for prediction"""
    df = pd.DataFrame([input_data])

    # 1. Calculate Distance if coordinates provided
    if all(k in input_data for k in ['restaurant_lat', 'restaurant_lon', 'delivery_lat', 'delivery_lon']):
        df['Distance_km'] = np.sqrt(
            (input_data['restaurant_lat'] - input_data['delivery_lat'])**2 +
            (input_data['restaurant_lon'] - input_data['delivery_lon'])**2
        ) * 111

    # 2. Time Features
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

    # 3. Age & Rating Features
    age = int(input_data.get('delivery_person_age', 30))
    df['Age_group'] = 'Young' if age <= 25 else 'Middle' if age <= 35 else 'Senior'

    rating = float(input_data.get('delivery_person_ratings', 4.5))
    df['Rating_category'] = 'Average' if rating <= 4.0 else 'Good' if rating <= 4.5 else 'Excellent'

    # 4. Map inputs to Feature Columns (Ensure exact match with training)
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

    # 5. Label Encoding
    for col in X.select_dtypes(include=['object']).columns:
        if col in label_encoders:
            try:
                X[col] = label_encoders[col].transform(X[col].astype(str))
            except:
                # Handle unseen categories
                X[col] = 0 

    return X

@app.route('/')
def home():
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        
        # Extract Prep Time for logic check
        prep_time = int(input_data.get('preparation_time', 15))
        
        X = prepare_input_features(input_data)
        prediction = model.predict(X)[0]
        
        # LOGIC FIX: Delivery cannot happen before preparation + transit
        # We assume at least 5 mins of travel time
        min_logical_time = prep_time + 5
        
        prediction = max(min_logical_time, prediction)
        prediction = min(120, prediction) # Cap at 2 hours to avoid outliers

        uncertainty = 3 # +/- minutes range
        predicted_time = float(round(prediction, 1))
        
        return jsonify({
            'success': True,
            'predicted_time': predicted_time,
            'predicted_time_min': float(round(prediction - uncertainty, 1)),
            'predicted_time_max': float(round(prediction + uncertainty, 1)),
            'message': f'Estimated delivery time: {int(predicted_time)} minutes'
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': model_info['best_model_name']})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
