import pandas as pd
import numpy as np

def engineer_features(df):
    """
    Takes the cleaned dataframe and adds new features for analysis.
    """
    # Create a copy to avoid SettingWithCopy warnings
    df_feat = df.copy()

    # 1. Extract Hour from DateTime
    df_feat['hour'] = df_feat['date_time'].dt.hour

    # 2. Create Temperature in Celsius
    df_feat['temp_celsius'] = df_feat['temp'] - 273.15

    # 3. Create Traffic Category using conditionals
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
