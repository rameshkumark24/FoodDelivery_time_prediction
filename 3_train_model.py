import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("FOOD DELIVERY TIME PREDICTION - MODEL TRAINING")
print("="*70)

# Load processed data
print("\nLoading processed data...")
df = pd.read_csv('food_delivery_processed.csv')
print(f"Dataset shape: {df.shape}")

# ==================== DATA PREPARATION ====================
print("\n" + "="*70)
print("DATA PREPARATION")
print("="*70)

def prepare_features(df):
    """Prepare features for modeling"""
    
    df_model = df.copy()
    
    # Select features for modeling
    feature_columns = [
        'Distance_km',
        'Delivery_person_Age',
        'Delivery_person_Ratings',
        'Preparation_time_min',
        'Order_hour',
        'Day_of_week',
        'Month',
        'Is_weekend',
        'Is_peak_hour',
        'Weather_conditions',
        'Road_traffic_density',
        'Type_of_vehicle',
        'Type_of_order',
        'Festival',
        'City',
        'Time_period',
        'Age_group',
        'Rating_category'
    ]
    
    target_column = 'Time_taken_min'
    
    # Create feature dataframe
    X = df_model[feature_columns].copy()
    y = df_model[target_column].copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    print(f"\nEncoding {len(categorical_features)} categorical features...")
    for col in categorical_features:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    return X, y, label_encoders, feature_columns

X, y, label_encoders, feature_columns = prepare_features(df)

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nFeature list ({len(feature_columns)} features):")
for i, feature in enumerate(feature_columns, 1):
    print(f"  {i}. {feature}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n" + "="*70)
print("SPLITTING DATA")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ==================== MODEL TRAINING ====================
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

models = {
    'Random Forest': RandomForestRegressor(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.1,
        random_state=42
    ),
    'XGBoost': XGBRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=200,
        max_depth=7,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*70}")
    print(f"Training {name}...")
    print(f"{'='*70}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Calculate accuracy (within ±5 minutes)
    train_accuracy = np.mean(np.abs(y_train - y_train_pred) <= 5) * 100
    test_accuracy = np.mean(np.abs(y_test - y_test_pred) <= 5) * 100
    
    results[name] = {
        'model': model,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'y_test_pred': y_test_pred
    }
    
    print(f"\n{name} Performance:")
    print(f"  Training MAE: {train_mae:.2f} minutes")
    print(f"  Test MAE: {test_mae:.2f} minutes")
    print(f"  Training RMSE: {train_rmse:.2f} minutes")
    print(f"  Test RMSE: {test_rmse:.2f} minutes")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Training Accuracy (±5 min): {train_accuracy:.2f}%")
    print(f"  Test Accuracy (±5 min): {test_accuracy:.2f}%")

# ==================== MODEL COMPARISON ====================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test MAE': [results[m]['test_mae'] for m in results.keys()],
    'Test RMSE': [results[m]['test_rmse'] for m in results.keys()],
    'Test R²': [results[m]['test_r2'] for m in results.keys()],
    'Test Accuracy (±5min)': [results[m]['test_accuracy'] for m in results.keys()]
})

comparison_df = comparison_df.sort_values('Test MAE')
print("\n", comparison_df.to_string(index=False))

# Select best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   Test MAE: {results[best_model_name]['test_mae']:.2f} minutes")
print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"   Accuracy (±5 min): {results[best_model_name]['test_accuracy']:.2f}%")

# ==================== FEATURE IMPORTANCE ====================
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['Feature'].head(15), 
             feature_importance['Importance'].head(15),
             color='steelblue')
    plt.xlabel('Importance')
    plt.title(f'Top 15 Feature Importances - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('visualizations/06_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✅ Feature importance plot saved")

# ==================== PREDICTION ANALYSIS ====================
print("\n" + "="*70)
print("PREDICTION ANALYSIS")
print("="*70)

y_test_pred = results[best_model_name]['y_test_pred']

# Actual vs Predicted plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_test_pred, alpha=0.5, s=20)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Delivery Time (minutes)')
plt.ylabel('Predicted Delivery Time (minutes)')
plt.title(f'Actual vs Predicted - {best_model_name}')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_test_pred
plt.hist(residuals, bins=50, color='lightcoral', edgecolor='black')
plt.xlabel('Prediction Error (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Prediction Errors')
plt.axvline(0, color='red', linestyle='--', linewidth=2)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/07_prediction_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Prediction analysis plot saved")

# Error analysis
print("\nError Analysis:")
print(f"  Mean Error: {residuals.mean():.2f} minutes")
print(f"  Std Error: {residuals.std():.2f} minutes")
print(f"  Predictions within ±3 min: {np.mean(np.abs(residuals) <= 3) * 100:.2f}%")
print(f"  Predictions within ±5 min: {np.mean(np.abs(residuals) <= 5) * 100:.2f}%")
print(f"  Predictions within ±10 min: {np.mean(np.abs(residuals) <= 10) * 100:.2f}%")

# ==================== SAVE MODELS ====================
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

import os
os.makedirs('models', exist_ok=True)

# Save best model
joblib.dump(best_model, 'models/best_model.pkl')
print(f"\n✅ Best model ({best_model_name}) saved as 'models/best_model.pkl'")

# Save label encoders
joblib.dump(label_encoders, 'models/label_encoders.pkl')
print("✅ Label encoders saved as 'models/label_encoders.pkl'")

# Save feature columns
joblib.dump(feature_columns, 'models/feature_columns.pkl')
print("✅ Feature columns saved as 'models/feature_columns.pkl'")

# Save all models
for name, result in results.items():
    model_filename = f"models/{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(result['model'], model_filename)
print(f"\n✅ All {len(results)} models saved in 'models/' folder")

# Save model comparison
comparison_df.to_csv('models/model_comparison.csv', index=False)
print("✅ Model comparison saved as 'models/model_comparison.csv'")

# ==================== CREATE MODEL INFO ====================
model_info = {
    'best_model_name': best_model_name,
    'test_mae': results[best_model_name]['test_mae'],
    'test_rmse': results[best_model_name]['test_rmse'],
    'test_r2': results[best_model_name]['test_r2'],
    'test_accuracy': results[best_model_name]['test_accuracy'],
    'feature_columns': feature_columns,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

joblib.dump(model_info, 'models/model_info.pkl')
print("✅ Model info saved as 'models/model_info.pkl'")

print("\n" + "="*70)
print("MODEL TRAINING COMPLETE!")
print("="*70)
print(f"\n📊 Summary:")
print(f"   Best Model: {best_model_name}")
print(f"   Test MAE: {results[best_model_name]['test_mae']:.2f} minutes")
print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
print(f"   Accuracy (±5 min): {results[best_model_name]['test_accuracy']:.2f}%")
print(f"\n📁 All files saved in 'models/' folder")
print(f"📊 Visualizations saved in 'visualizations/' folder")