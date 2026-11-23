import logging
import sys

# Set up logger at the top of every script
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

logger.info("Loading data...")
logger.info("--- K-Means Clustering ---")

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
print("Loading data...")
file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'
df = pd.read_csv(file_path)
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.sort_values('date_time').reset_index(drop=True)

# Feature Engineering (Time)
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Select features for the models
features = ['hour', 'day_of_week', 'is_weekend', 'temp', 'rain_1h', 'clouds_all']
target = 'traffic_volume'

# Scale the data (LSTMs require scaled data between 0 and 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[features + [target]])

# ==========================================
# TASK 3.1: LSTM for Demand Prediction
# ==========================================
print("\nTraining LSTM (This will take a minute)...")

# Create sequences (using past 24 hours to predict the next hour's traffic)
X_lstm, y_lstm = [], []
time_steps = 24 
for i in range(time_steps, len(scaled_data)):
    X_lstm.append(scaled_data[i-time_steps:i, :-1]) # All features except target
    y_lstm.append(scaled_data[i, -1])               # The target (traffic volume)

X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

# Train/Test Split (80% train, 20% test)
split_idx = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:split_idx], X_lstm[split_idx:]
y_train, y_test = y_lstm[:split_idx], y_lstm[split_idx:]

# Build the LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=64, epochs=3, validation_data=(X_test, y_test), verbose=1)

# Evaluate LSTM
lstm_pred = model.predict(X_test)
# Reverse the scaling to get real traffic volume numbers
dummy_array = np.zeros((len(lstm_pred), len(features) + 1))
dummy_array[:, -1] = lstm_pred.flatten()
lstm_pred_actual = scaler.inverse_transform(dummy_array)[:, -1]

y_test_actual = df[target].iloc[-len(y_test):].values
mae_lstm = np.mean(np.abs(y_test_actual - lstm_pred_actual))
print(f"LSTM Mean Absolute Error (MAE): {mae_lstm:.2f} vehicles")


# ==========================================
# TASK 3.2: Explainability with SHAP (Using Proxy Model)
# ==========================================
print("\nTraining Proxy Random Forest for SHAP Explanations...")

# Train a Random Forest on the exact same features (flattened for tabular model)
X_rf = df[features]
y_rf = df[target]

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_rf, y_rf, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Calculate SHAP values using the TreeExplainer
explainer = shap.TreeExplainer(rf_model)
# Take a random sample of 500 rows so the SHAP plot generates fast
X_sample = X_test_rf.sample(500, random_state=42)
shap_values = explainer.shap_values(X_sample)

# Print global feature importance
print("\nSHAP Feature Importance (Mean Absolute SHAP Value):")
mean_shap = np.abs(shap_values).mean(axis=0)
for feat, val in zip(features, mean_shap):
    print(f"{feat}: {val:.2f}")

# Generate a Summary Plot and save it
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig('shap_summary_plot.png', bbox_inches='tight')
print("\nSHAP summary plot saved as 'shap_summary_plot.png' in your folder!")
