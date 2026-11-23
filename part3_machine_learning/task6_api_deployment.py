from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import warnings
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

app = FastAPI(title="Traffic Prediction API")

logger.info("Training model for deployment...")
file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'
df = pd.read_csv(file_path)
df['date_time'] = pd.to_datetime(df['date_time'])
df['hour'] = df['date_time'].dt.hour
df['day_of_week'] = df['date_time'].dt.day_of_week

features = ['hour', 'day_of_week', 'temp', 'rain_1h', 'clouds_all']
X = df[features]
y = df['traffic_volume']

model = RandomForestRegressor(n_estimators=20, random_state=42)
model.fit(X, y)
logger.info("Model ready!")

class TrafficInput(BaseModel):
    hour: int
    day_of_week: int
    temp: float
    rain_1h: float
    clouds_all: int

@app.post("/predict")
def predict_traffic(data: TrafficInput):
    input_df = pd.DataFrame([{
        'hour': data.hour,
        'day_of_week': data.day_of_week,
        'temp': data.temp,
        'rain_1h': data.rain_1h,
        'clouds_all': data.clouds_all
    }])
    
    prediction = model.predict(input_df)[0]
    
    return {
        "predicted_traffic_volume": round(float(prediction), 2),
        "status": "success"
    }
