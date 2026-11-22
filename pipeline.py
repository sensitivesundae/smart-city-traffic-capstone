import pandas as pd
import numpy as np
import logging

# ==========================================
# Task 1: Mandatory Logging Configuration
# ==========================================
logging.basicConfig(
    filename='pipeline.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_pipeline():
    logging.info("Starting data pipeline.")
    
    # Update this path if you move the CSV file!
    file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'

    # ==========================================
    # 1. Load the raw CSV file
    # ==========================================
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
    except FileNotFoundError:
        logging.error(f"File '{file_path}' not found. Please check the path.")
        return None
    except pd.errors.EmptyDataError:
        logging.error("The file is empty.")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

    # ==========================================
    # 2. Validate the data schema
    # ==========================================
    expected_cols = [
        'holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 
        'weather_main', 'weather_description', 'date_time', 'traffic_volume'
    ]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if len(missing_cols) > 0:
        logging.error(f"Schema validation failed. Missing columns: {missing_cols}")
        return None
    logging.info("Schema validation passed. All expected columns are present.")

    # ==========================================
    # 3. Clean the data
    # ==========================================
    # Standardise categorical values
    df['weather_main'] = df['weather_main'].str.lower().str.strip()
    df['holiday'] = df['holiday'].str.lower().str.strip()
    
    # Parse date/time
    df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
    
    # Remove duplicates
    initial_len = len(df)
    df = df.drop_duplicates()
    logging.info(f"Removed {initial_len - len(df)} duplicate rows.")

    # Detect impossible outliers and replace with NaN
    df.loc[df['temp'] <= 0, 'temp'] = np.nan 
    df.loc[df['rain_1h'] > 9000, 'rain_1h'] = np.nan 
    logging.info("Impossible values (0 Kelvin, >9000mm rain) converted to NaN.")

    # ==========================================
    # 4. Use conditionals and loops (Imputation by Month)
    # ==========================================
    df['month'] = df['date_time'].dt.month

    for month in range(1, 13):
        month_condition = df['month'] == month
        
        # Calculate medians for the specific month
        month_temp_median = df.loc[month_condition, 'temp'].median()
        month_rain_median = df.loc[month_condition, 'rain_1h'].median()
        
        # Impute missing values for that month
        df.loc[month_condition, 'temp'] = df.loc[month_condition, 'temp'].fillna(month_temp_median)
        df.loc[month_condition, 'rain_1h'] = df.loc[month_condition, 'rain_1h'].fillna(month_rain_median)

    df = df.drop(columns=['month'])
    logging.info("Imputation complete using monthly medians.")
    logging.info("Pipeline finished successfully.")
    
    return df

if __name__ == "__main__":
    cleaned_data = run_pipeline()
    if cleaned_data is not None:
        print("Pipeline ran successfully. Check pipeline.log for details.")
        # Optional: Save the cleaned data to a new CSV
        # cleaned_data.to_csv('cleaned_traffic_data.csv', index=False)
