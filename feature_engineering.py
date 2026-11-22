import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Takes the cleaned dataframe and adds new features for ML.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df_feat = df.copy()

    # ==========================================
    # 1. Time Features
    # ==========================================
    df_feat['hour'] = df_feat['date_time'].dt.hour
    df_feat['day_of_week'] = df_feat['date_time'].dt.dayofweek # 0=Monday, 6=Sunday
    
    # Weekend indicator (1 if Saturday/Sunday, else 0)
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for Hour (captures the circular nature of a 24-hour clock)
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)

    # ==========================================
    # 2. Weather Features
    # ==========================================
    # Create derived indicators
    df_feat['is_raining'] = (df_feat['rain_1h'] > 0).astype(int)
    df_feat['is_snowing'] = (df_feat['snow_1h'] > 0).astype(int)
    
    # Encode categorical weather (One-Hot Encoding)
    # This turns 'weather_main' into columns like 'weather_clear', 'weather_clouds', etc.
    df_feat = pd.get_dummies(df_feat, columns=['weather_main'], drop_first=True)

    # ==========================================
    # 3. Numerical Features (Normalisation)
    # ==========================================
    # Normalise at least two continuous variables (Temp and Clouds_all)
    # Formula: (X - Min) / (Max - Min)
    
    # Normalise Temperature
    temp_min = df_feat['temp'].min()
    temp_max = df_feat['temp'].max()
    df_feat['temp_normalized'] = (df_feat['temp'] - temp_min) / (temp_max - temp_min)
    
    # Normalise Clouds
    clouds_min = df_feat['clouds_all'].min()
    clouds_max = df_feat['clouds_all'].max()
    df_feat['clouds_normalized'] = (df_feat['clouds_all'] - clouds_min) / (clouds_max - clouds_min)

    # ==========================================
    # 4. Traffic Target
    # ==========================================
    # Data-driven congestion category based on traffic volume
    # Logic: <4500 = Low, 4500-5500 = Medium, >5500 = High
    conditions = [
        (df_feat['traffic_volume'] < 4500),
        (df_feat['traffic_volume'] >= 4500) & (df_feat['traffic_volume'] <= 5500),
        (df_feat['traffic_volume'] > 5500)
    ]
    choices = ['Low', 'Medium', 'High']
    
    df_feat['traffic_category'] = np.select(conditions, choices, default='Unknown')

    return df_feat

if __name__ == "__main__":
    print("Feature engineering module loaded. Use engineer_features(df) to apply.")
