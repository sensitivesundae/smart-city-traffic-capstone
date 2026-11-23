import pandas as pd
import logging
import sys

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)

def get_travel_recommendation(df, is_weekend, weather_condition):
    df = df.copy()
    df['is_weekend'] = (df['date_time'].dt.day_of_week >= 5)
    df['hour'] = df['date_time'].dt.hour
    df['weather_main'] = df['weather_main'].str.lower().str.strip()
    
    filtered_df = df[df['is_weekend'] == is_weekend]
    
    if weather_condition.lower() != 'any':
        filtered_df = filtered_df[filtered_df['weather_main'] == weather_condition.lower()]
        
    if filtered_df.empty:
        return "Sorry, not enough data to make a recommendation for those conditions."

    hourly_avg = filtered_df.groupby('hour')['traffic_volume'].mean()
    lowest_hours = hourly_avg.nsmallest(2).index.tolist()
    lowest_hours.sort()
    
    day_str = "weekend" if is_weekend else "weekday"
    weather_str = f"{weather_condition.lower()} weather" if weather_condition.lower() != 'any' else "current weather conditions"
    
    start_time = pd.to_datetime(f"{lowest_hours[0]}:00", format="%H:%M").strftime("%I:%M %p")
    end_time = pd.to_datetime(f"{lowest_hours[-1] + 2}:00", format="%H:%M").strftime("%I:%M %p")
    
    avg_vol = hourly_avg[lowest_hours].mean()

    recommendation = (f"For a {day_str} journey during {weather_str}, "
                      f"consider travelling between {start_time} and {end_time}, "
                      f"when historical traffic volumes are typically lower "
                      f"(average of {avg_vol:.0f} vehicles/hour).")
                      
    return recommendation

if __name__ == "__main__":
    logger.info("Loading data...")
    file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'
    df = pd.read_csv(file_path)
    df['date_time'] = pd.to_datetime(df['date_time'])

    logger.info("--- TRAFFIC RECOMMENDATION SYSTEM ---")
    
    logger.info("Scenario 1:")
    # print() is allowed here because this is the direct output for the end-user
    print(get_travel_recommendation(df, is_weekend=False, weather_condition='clear'))
    
    logger.info("Scenario 2:")
    print(get_travel_recommendation(df, is_weekend=True, weather_condition='rain'))
    
    logger.info("Scenario 3:")
    print(get_travel_recommendation(df, is_weekend=False, weather_condition='any'))
