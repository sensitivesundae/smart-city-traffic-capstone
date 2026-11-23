import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
import pandas as pd

def get_travel_recommendation(df, is_weekend, weather_condition):
    """
    Analyzes historical data to find the best (lowest traffic) hour to travel.
    """
    # Create a copy to prevent pandas slicing errors
    df = df.copy()
    
    # 1. Create the columns FIRST before doing any filtering
    # (I also fixed the deprecation warning by changing dayofweek to day_of_week)
    df['is_weekend'] = (df['date_time'].dt.day_of_week >= 5)
    df['hour'] = df['date_time'].dt.hour
    df['weather_main'] = df['weather_main'].str.lower().str.strip()
    
    # 2. Filter data based on user inputs
    filtered_df = df[df['is_weekend'] == is_weekend]
    
    if weather_condition.lower() != 'any':
        filtered_df = filtered_df[filtered_df['weather_main'] == weather_condition.lower()]
        
    if filtered_df.empty:
        return "Sorry, not enough data to make a recommendation for those conditions."

    # 3. Group by hour and find the average traffic volume
    hourly_avg = filtered_df.groupby('hour')['traffic_volume'].mean()
    
    # 4. Identify the lowest traffic period
    # We take the top 2 lowest hours to give a 2-hour window
    lowest_hours = hourly_avg.nsmallest(2).index.tolist()
    lowest_hours.sort() # Ensure they are in chronological order
    
    # 5. Generate plain-language recommendation
    day_str = "weekend" if is_weekend else "weekday"
    weather_str = f"{weather_condition.lower()} weather" if weather_condition.lower() != 'any' else "current weather conditions"
    
    # Format time nicely (e.g., 14 -> 2:00 PM)
    start_time = pd.to_datetime(f"{lowest_hours[0]}:00", format="%H:%M").strftime("%I:%M %p")
    end_time = pd.to_datetime(f"{lowest_hours[-1] + 2}:00", format="%H:%M").strftime("%I:%M %p")
    
    avg_vol = hourly_avg[lowest_hours].mean()

    recommendation = (f"For a {day_str} journey during {weather_str}, "
                      f"consider travelling between {start_time} and {end_time}, "
                      f"when historical traffic volumes are typically lower "
                      f"(average of {avg_vol:.0f} vehicles/hour).")
                      
    return recommendation

# ==========================================
# RUN THE SYSTEM
# ==========================================
if __name__ == "__main__":
    print("Loading data...")
    file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'
    df = pd.read_csv(file_path)
    df['date_time'] = pd.to_datetime(df['date_time'])

    print("\n--- TRAFFIC RECOMMENDATION SYSTEM ---\n")
    
    # Test 1: Weekday, Clear Weather
    print("Scenario 1:")
    print(get_travel_recommendation(df, is_weekend=False, weather_condition='clear'))
    
    print("\nScenario 2:")
    # Test 2: Weekend, Rain
    print(get_travel_recommendation(df, is_weekend=True, weather_condition='rain'))
    
    print("\nScenario 3:")
    # Test 3: Weekday, Any Weather
    print(get_travel_recommendation(df, is_weekend=False, weather_condition='any'))
