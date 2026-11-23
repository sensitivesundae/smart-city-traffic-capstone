import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

print("Loading data and training baseline model...")
file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'
df = pd.read_csv(file_path)
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.day_of_week
features = ['hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all']

# Train baseline model on first 80%
train = df.iloc[:int(len(df)*0.8)]
X_train, y_train = train[features], train['traffic_volume']
model = RandomForestRegressor(n_estimators=20, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# TASK 6.4 & 6.5: MONITORING & ALERTING SIMULATION
# ==========================================
print("\n--- RUNNING MONITORING DASHBOARD ---")

# Simulate new incoming data (the last 20% of the dataset)
new_data = df.iloc[int(len(df)*0.8):].sample(500, random_state=42) 
X_new, y_actual = new_data[features], new_data['traffic_volume']

# 1. Check for Prediction Error Drift
y_pred = model.predict(X_new)
current_mae = mean_absolute_error(y_actual, y_pred)
baseline_mae = 1800 # Hypothetical baseline from training phase
error_threshold = baseline_mae * 1.20 # Alert if error increases by 20%

# 2. Check for Feature Distribution Drift (e.g., Temperature)
train_mean_temp = X_train['temp'].mean()
new_mean_temp = X_new['temp'].mean()
temp_drift_threshold = 5.0 # Alert if average temp shifts by 5 Kelvin

print(f"Current MAE: {current_mae:.2f} (Threshold: {error_threshold:.2f})")
print(f"Train Avg Temp: {train_mean_temp:.2f}K | New Data Avg Temp: {new_mean_temp:.2f}K")

# 3. Alerting Mechanism
alert_triggered = False

if current_mae > error_threshold:
    print(f"[ALERT] Prediction Error Drift detected! MAE jumped to {current_mae:.2f}")
    alert_triggered = True
    
if abs(new_mean_temp - train_mean_temp) > temp_drift_threshold:
    print(f"[ALERT] Feature Distribution Drift detected! Temperature shifted by {abs(new_mean_temp - train_mean_temp):.2f}K")
    alert_triggered = True

print("\n--- SYSTEM STATUS ---")
if alert_triggered:
    print("🔴 STATUS: ALERT / Requires Investigation")
else:
    print("🟢 STATUS: PASS / Normal")
