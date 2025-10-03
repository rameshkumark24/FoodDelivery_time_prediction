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
model = joblib.load('models/best_model.pkl')
label_encoders = joblib.load('models/label_encoders.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')
model_info = joblib.load('models/model_info.pkl')
print(f"✅ Model loaded: {model_info['best_model_name']}")
print(f"✅ Test MAE: {model_info['test_mae']:.2f} minutes")

def prepare_input_features(input_data):
    """Prepare input data for prediction"""
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

    if 6 <= order_hour < 12:
        df['Time_period'] = 'Morning'
    elif 12 <= order_hour < 17:
        df['Time_period'] = 'Afternoon'
    elif 17 <= order_hour < 21:
        df['Time_period'] = 'Evening'
    else:
        df['Time_period'] = 'Night'

    age = input_data.get('delivery_person_age', 30)
    if age <= 25:
        df['Age_group'] = 'Young'
    elif age <= 35:
        df['Age_group'] = 'Middle'
    else:
        df['Age_group'] = 'Senior'

    rating = input_data.get('delivery_person_ratings', 4.5)
    if rating <= 4.0:
        df['Rating_category'] = 'Average'
    elif rating <= 4.5:
        df['Rating_category'] = 'Good'
    else:
        df['Rating_category'] = 'Excellent'

    X = pd.DataFrame(columns=feature_columns)

    feature_mapping = {
        'Distance_km': df['Distance_km'].values[0] if 'Distance_km' in df else input_data.get('distance_km', 5),
        'Delivery_person_Age': input_data.get('delivery_person_age', 30),
        'Delivery_person_Ratings': input_data.get('delivery_person_ratings', 4.5),
        'Preparation_time_min': input_data.get('preparation_time', 20),
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
        X = prepare_input_features(input_data)
        prediction = model.predict(X)[0]
        prediction = max(15, min(90, prediction))
        uncertainty = 3

        predicted_time = float(round(prediction, 1))
        predicted_time_min = float(round(prediction - uncertainty, 1))
        predicted_time_max = float(round(prediction + uncertainty, 1))

        response = {
            'success': True,
            'predicted_time': predicted_time,
            'predicted_time_min': predicted_time_min,
            'predicted_time_max': predicted_time_max,
            'message': f'Estimated delivery time: {int(predicted_time)} minutes'
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/model-info')
def get_model_info():
    return jsonify({
        'model_name': model_info['best_model_name'],
        'test_mae': round(model_info['test_mae'], 2),
        'test_r2': round(model_info['test_r2'], 4),
        'test_accuracy': round(model_info['test_accuracy'], 2),
        'training_date': model_info['training_date']
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': True})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*70)
    print("🚀 FOOD DELIVERY TIME PREDICTION API")
    print("="*70)
    print(f"Model: {model_info['best_model_name']}")
    print(f"Test MAE: {model_info['test_mae']:.2f} minutes")
    print(f"Test R²: {model_info['test_r2']:.4f}")
    print(f"Accuracy (±5 min): {model_info['test_accuracy']:.2f}%")
    print("="*70)
    print("\n🌐 Starting Flask server...")
    print(f"📱 Access the app at: http://0.0.0.0:{port}")
    print("\n")
    app.run(host='0.0.0.0', port=port)
