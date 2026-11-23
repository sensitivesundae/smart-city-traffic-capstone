import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 1. Load and Prep Data (Simplified for this task)
print("Loading data...")
file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'
df = pd.read_csv(file_path)
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek

features = ['hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all']
X = df[features]
y = df['traffic_volume']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Set up MLflow Experiment
mlflow.set_experiment("Traffic_Volume_Experiments")

# 3. Experiment A: Random Forest (50 trees)
print("Running Experiment A...")
with mlflow.start_run(run_name="RF_50_Trees"):
    n_est = 50
    rf_a = RandomForestRegressor(n_estimators=n_est, random_state=42)
    rf_a.fit(X_train, y_train)
    preds_a = rf_a.predict(X_test)
    mae_a = mean_absolute_error(y_test, preds_a)
    
    # Log parameters and metrics to MLflow
    mlflow.log_param("n_estimators", n_est)
    mlflow.log_metric("MAE", mae_a)
    # Save the model itself
    mlflow.sklearn.log_model(rf_a, "model")

# 4. Experiment B: Random Forest (100 trees)
print("Running Experiment B...")
with mlflow.start_run(run_name="RF_100_Trees"):
    n_est = 100
    rf_b = RandomForestRegressor(n_estimators=n_est, random_state=42)
    rf_b.fit(X_train, y_train)
    preds_b = rf_b.predict(X_test)
    mae_b = mean_absolute_error(y_test, preds_b)
    
    # Log parameters and metrics to MLflow
    mlflow.log_param("n_estimators", n_est)
    mlflow.log_metric("MAE", mae_b)
    mlflow.sklearn.log_model(rf_b, "model")

print("Done! Experiments logged. Run 'mlflow ui' in your terminal to view them.")
