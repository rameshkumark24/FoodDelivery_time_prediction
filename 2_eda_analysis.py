import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
print("Loading dataset...")
df = pd.read_csv('food_delivery_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

# ==================== DATA EXPLORATION ====================
print("\n" + "="*60)
print("DATA EXPLORATION")
print("="*60)

print("\nMissing values:")
print(df.isnull().sum())

print("\nData types:")
print(df.dtypes)

print("\nUnique values in categorical columns:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col not in ['Order_ID', 'Delivery_person_ID', 'Order_Date', 'Time_Orderd', 'Time_Order_picked']:
        print(f"{col}: {df[col].nunique()} unique values - {df[col].unique()}")

# ==================== FEATURE ENGINEERING ====================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

def feature_engineering(df):
    """Create new features from existing data"""
    
    df_fe = df.copy()
    
    # 1. Extract time features
    df_fe['Order_Date'] = pd.to_datetime(df_fe['Order_Date'])
    df_fe['Day_of_week'] = df_fe['Order_Date'].dt.dayofweek
    df_fe['Day_name'] = df_fe['Order_Date'].dt.day_name()
    df_fe['Month'] = df_fe['Order_Date'].dt.month
    df_fe['Is_weekend'] = (df_fe['Day_of_week'] >= 5).astype(int)
    
    # 2. Extract hour from time
    df_fe['Order_hour'] = pd.to_datetime(df_fe['Time_Orderd'], format='%H:%M').dt.hour
    df_fe['Pickup_hour'] = pd.to_datetime(df_fe['Time_Order_picked'], format='%H:%M').dt.hour
    
    # 3. Create time period features
    def get_time_period(hour):
        if 6 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df_fe['Time_period'] = df_fe['Order_hour'].apply(get_time_period)
    
    # 4. Is peak hour (lunch: 12-14, dinner: 19-21)
    df_fe['Is_peak_hour'] = df_fe['Order_hour'].apply(
        lambda x: 1 if (12 <= x <= 14) or (19 <= x <= 21) else 0
    )
    
    # 5. Preparation time (difference between order and pickup)
    order_time = pd.to_datetime(df_fe['Time_Orderd'], format='%H:%M')
    pickup_time = pd.to_datetime(df_fe['Time_Order_picked'], format='%H:%M')
    df_fe['Preparation_time_min'] = ((pickup_time - order_time).dt.total_seconds() / 60).abs()
    
    # 6. Distance categories
    df_fe['Distance_category'] = pd.cut(
        df_fe['Distance_km'],
        bins=[0, 3, 6, 10, float('inf')],
        labels=['Very_Close', 'Close', 'Moderate', 'Far']
    )
    
    # 7. Age groups
    df_fe['Age_group'] = pd.cut(
        df_fe['Delivery_person_Age'],
        bins=[0, 25, 35, 100],
        labels=['Young', 'Middle', 'Senior']
    )
    
    # 8. Rating categories
    df_fe['Rating_category'] = pd.cut(
        df_fe['Delivery_person_Ratings'],
        bins=[0, 4.0, 4.5, 5.0],
        labels=['Average', 'Good', 'Excellent']
    )
    
    # 9. Experience level (based on ID - assuming lower IDs are more experienced)
    df_fe['Delivery_person_ID_num'] = df_fe['Delivery_person_ID'].str.extract('(\d+)').astype(int)
    df_fe['Experience_level'] = pd.qcut(
        df_fe['Delivery_person_ID_num'],
        q=3,
        labels=['Experienced', 'Intermediate', 'New']
    )
    
    return df_fe

df_processed = feature_engineering(df)

print(f"\n✅ Feature engineering complete!")
print(f"Original features: {df.shape[1]}")
print(f"New features: {df_processed.shape[1]}")
print(f"\nNew features created:")
new_features = set(df_processed.columns) - set(df.columns)
for feature in sorted(new_features):
    print(f"  - {feature}")

# ==================== VISUALIZATIONS ====================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# Create output directory
import os
os.makedirs('visualizations', exist_ok=True)

# 1. Target variable distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df_processed['Time_taken_min'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Delivery Time (minutes)')
plt.ylabel('Frequency')
plt.title('Distribution of Delivery Time')
plt.axvline(df_processed['Time_taken_min'].mean(), color='red', linestyle='--', label=f'Mean: {df_processed["Time_taken_min"].mean():.1f} min')
plt.legend()

plt.subplot(1, 2, 2)
plt.boxplot(df_processed['Time_taken_min'])
plt.ylabel('Delivery Time (minutes)')
plt.title('Delivery Time Box Plot')
plt.tight_layout()
plt.savefig('visualizations/01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Delivery time by categorical features
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

categorical_features = ['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle', 
                        'Type_of_order', 'Time_period', 'City']

for idx, feature in enumerate(categorical_features):
    ax = axes[idx // 3, idx % 3]
    df_processed.groupby(feature)['Time_taken_min'].mean().sort_values().plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Average Delivery Time (min)')
    ax.set_title(f'Impact of {feature}')
    ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/02_categorical_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Correlation with numerical features
plt.figure(figsize=(10, 8))
numerical_features = ['Distance_km', 'Delivery_person_Age', 'Delivery_person_Ratings', 
                     'Preparation_time_min', 'Order_hour', 'Time_taken_min']
correlation_matrix = df_processed[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
            square=True, linewidths=1)
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('visualizations/03_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Time analysis
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# By hour
df_processed.groupby('Order_hour')['Time_taken_min'].mean().plot(ax=axes[0], marker='o', color='green')
axes[0].set_xlabel('Hour of Day')
axes[0].set_ylabel('Average Delivery Time (min)')
axes[0].set_title('Delivery Time by Hour')
axes[0].grid(alpha=0.3)

# By day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df_processed.groupby('Day_name')['Time_taken_min'].mean().reindex(day_order).plot(ax=axes[1], marker='o', color='purple')
axes[1].set_xlabel('Day of Week')
axes[1].set_ylabel('Average Delivery Time (min)')
axes[1].set_title('Delivery Time by Day of Week')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(alpha=0.3)

# Peak vs Non-peak
df_processed.groupby('Is_peak_hour')['Time_taken_min'].mean().plot(kind='bar', ax=axes[2], color=['lightblue', 'salmon'])
axes[2].set_xlabel('Peak Hour')
axes[2].set_ylabel('Average Delivery Time (min)')
axes[2].set_title('Peak Hour Impact')
axes[2].set_xticklabels(['Non-Peak', 'Peak'], rotation=0)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/04_time_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Distance vs Time scatter
plt.figure(figsize=(12, 6))
for weather in df_processed['Weather_conditions'].unique():
    data = df_processed[df_processed['Weather_conditions'] == weather]
    plt.scatter(data['Distance_km'], data['Time_taken_min'], alpha=0.5, label=weather, s=30)
plt.xlabel('Distance (km)')
plt.ylabel('Delivery Time (minutes)')
plt.title('Distance vs Delivery Time (colored by Weather)')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('visualizations/05_distance_vs_time.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ All visualizations saved in 'visualizations/' folder")

# ==================== SAVE PROCESSED DATA ====================
df_processed.to_csv('food_delivery_processed.csv', index=False)
print(f"\n✅ Processed data saved as 'food_delivery_processed.csv'")

# ==================== SUMMARY STATISTICS ====================
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)

print("\nDelivery Time Statistics:")
print(f"Mean: {df_processed['Time_taken_min'].mean():.2f} minutes")
print(f"Median: {df_processed['Time_taken_min'].median():.2f} minutes")
print(f"Std Dev: {df_processed['Time_taken_min'].std():.2f} minutes")
print(f"Min: {df_processed['Time_taken_min'].min():.0f} minutes")
print(f"Max: {df_processed['Time_taken_min'].max():.0f} minutes")

print("\nTop factors affecting delivery time:")
for feature in ['Weather_conditions', 'Road_traffic_density', 'Type_of_vehicle']:
    print(f"\n{feature}:")
    print(df_processed.groupby(feature)['Time_taken_min'].mean().sort_values(ascending=False))

print("\n" + "="*60)
print("EDA COMPLETE!")
print("="*60)