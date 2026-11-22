import pandas as pd
import numpy as np
import logging

# Set up a logger for this specific module
logger = logging.getLogger(__name__)

def engineer_features(df):
    """
    Takes the cleaned dataframe and adds new features for ML.
    """
    # --- REQUIREMENT 1: Log dataset shape BEFORE ---
    logger.info(f"Starting feature engineering. Initial dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Create a copy to avoid SettingWithCopy warnings
    df_feat = df.copy()

    # ==========================================
    # 1. Time Features
    # ==========================================
    df_feat['hour'] = df_feat['date_time'].dt.hour
    df_feat['day_of_week'] = df_feat['date_time'].dt.dayofweek 
    df_feat['is_weekend'] = (df_feat['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding
    df_feat['hour_sin'] = np.sin(2 * np.pi * df_feat['hour'] / 24.0)
    df_feat['hour_cos'] = np.cos(2 * np.pi * df_feat['hour'] / 24.0)

    # ==========================================
    # 2. Weather Features
    # ==========================================
    df_feat['is_raining'] = (df_feat['rain_1h'] > 0).astype(int)
    df_feat['is_snowing'] = (df_feat['snow_1h'] > 0).astype(int)
    df_feat = pd.get_dummies(df_feat, columns=['weather_main'], drop_first=True)

    # ==========================================
    # 3. Numerical Features (Normalisation)
    # ==========================================
    temp_min = df_feat['temp'].min()
    temp_max = df_feat['temp'].max()
    clouds_min = df_feat['clouds_all'].min()
    clouds_max = df_feat['clouds_all'].max()
    
    # --- REQUIREMENT 2: Log intermediate values at DEBUG level ---
    logger.debug(f"Intermediate calculation - Temp Min: {temp_min}, Temp Max: {temp_max}")
    logger.debug(f"Intermediate calculation - Clouds Min: {clouds_min}, Clouds Max: {clouds_max}")
    
    df_feat['temp_normalized'] = (df_feat['temp'] - temp_min) / (temp_max - temp_min)
    df_feat['clouds_normalized'] = (df_feat['clouds_all'] - clouds_min) / (clouds_max - clouds_min)

    # ==========================================
    # 4. Traffic Target
    # ==========================================
    conditions = [
        (df_feat['traffic_volume'] < 4500),
        (df_feat['traffic_volume'] >= 4500) & (df_feat['traffic_volume'] <= 5500),
        (df_feat['traffic_volume'] > 5500)
    ]
    choices = ['Low', 'Medium', 'High']
    df_feat['traffic_category'] = np.select(conditions, choices, default='Unknown')

    # --- REQUIREMENT 1: Log dataset shape AFTER ---
    logger.info(f"Feature engineering complete. Final dataset shape: {df_feat.shape[0]} rows, {df_feat.shape[1]} columns.")

    return df_feat

if __name__ == "__main__":
    print("Feature engineering module loaded. Use engineer_features(df) to apply.")
