import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore') # To keep the output clean

# 1. Load Data
file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'
print("Loading data...")
df = pd.read_csv(file_path)
df['date_time'] = pd.to_datetime(df['date_time'])
df['weather_main'] = df['weather_main'].str.lower().str.strip()
df['holiday'] = df['holiday'].str.lower().str.strip()

# 2. Create Proxy Label (From your screenshot)
print("Creating proxy accident-risk label...")
q1, q2, q3 = df['traffic_volume'].quantile([0.25, 0.5, 0.75]).values
def bucket(v):
    if v <= q1: return "Low"
    elif v <= q2: return "Medium"
    elif v <= q3: return "High"
    return "Severe"
df['congestion_category'] = df['traffic_volume'].apply(bucket)

# Define severe weather and low visibility
SEVERE_WEATHER = ['rain', 'snow', 'thunderstorm', 'fog', 'mist', 'drizzle', 'squall']
df['is_low_visibility'] = df['weather_description'].str.contains('fog|mist|haze|smoke', case=False, na=False).astype(int)

high_congestion = df['congestion_category'].isin(["High", "Severe"])
risky_weather = df['weather_main'].isin(SEVERE_WEATHER) | (df['is_low_visibility'] == 1)
df['high_risk'] = (high_congestion & risky_weather).astype(int)

# 3. Feature Engineering (Time, Cyclical, Weather, Holiday)
print("Engineering features...")
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7.0)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7.0)

df['is_holiday'] = (df['holiday'] != 'none').astype(int)
df['is_raining'] = (df['rain_1h'] > 0).astype(int)
df['is_snowing'] = (df['snow_1h'] > 0).astype(int)

# One-hot encode weather_main
df = pd.get_dummies(df, columns=['weather_main'], drop_first=True)

# Define feature list (drop target columns and irrelevant text columns)
features = ['temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour', 'day_of_week', 'is_weekend', 
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_holiday', 'is_raining', 'is_snowing'] + \
           [col for col in df.columns if col.startswith('weather_main_')]

X = df[features]
y_class = df['high_risk']
y_reg = df['traffic_volume']

# 4. Train/Test Split
X_train, X_test, yc_train, yc_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
_, _, yr_train, yr_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)

print("\n" + "="*40)
print("CLASSIFICATION MODELS (Predicting High Risk)")
print("="*40)

# Model A: Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, yc_train)
yc_pred_lr = lr.predict(X_test)
yc_prob_lr = lr.predict_proba(X_test)[:, 1]

print("--- Logistic Regression Metrics ---")
print(f"Accuracy:  {accuracy_score(yc_test, yc_pred_lr):.4f}")
print(f"Precision: {precision_score(yc_test, yc_pred_lr):.4f}")
print(f"Recall:    {recall_score(yc_test, yc_pred_lr):.4f}")
print(f"F1-Score:  {f1_score(yc_test, yc_pred_lr):.4f}")
print(f"ROC AUC:   {roc_auc_score(yc_test, yc_prob_lr):.4f}")

# Model B: Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, yc_train)
yc_pred_rf = rf_clf.predict(X_test)
yc_prob_rf = rf_clf.predict_proba(X_test)[:, 1]

print("--- Random Forest Metrics ---")
print(f"Accuracy:  {accuracy_score(yc_test, yc_pred_rf):.4f}")
print(f"Precision: {precision_score(yc_test, yc_pred_rf):.4f}")
print(f"Recall:    {recall_score(yc_test, yc_pred_rf):.4f}")
print(f"F1-Score:  {f1_score(yc_test, yc_pred_rf):.4f}")
print(f"ROC AUC:   {roc_auc_score(yc_test, yc_prob_rf):.4f}")


print("\n" + "="*40)
print("REGRESSION MODELS (Predicting Traffic Volume)")
print("="*40)

# Model C: Linear Regression
print("\nTraining Linear Regression...")
lin_reg = LinearRegression()
lin_reg.fit(X_train, yr_train)
yr_pred_lin = lin_reg.predict(X_test)

print("--- Linear Regression Metrics ---")
print(f"MAE:  {mean_absolute_error(yr_test, yr_pred_lin):.2f}")
print(f"R-squared: {r2_score(yr_test, yr_pred_lin):.4f}")

# Model D: Random Forest Regressor
print("\nTraining Random Forest Regressor...")
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, yr_train)
yr_pred_rf = rf_reg.predict(X_test)

print("--- Random Forest Regressor Metrics ---")
print(f"MAE:  {mean_absolute_error(yr_test, yr_pred_rf):.2f}")
print(f"R-squared: {r2_score(yr_test, yr_pred_rf):.4f}")

print("\nDone!")
