import pandas as pd
import argparse
import logging
import sys
import os

# Set up logging
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Update this path if needed
CSV_PATH = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'

def main():
    # Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Mini Traffic Analytics App")
    parser.add_argument('command', choices=['query-date', 'high-traffic', 'compare-weekend'], 
                        help="The command to run.")
    parser.add_argument('--date', help="Date to query (format: YYYY-MM-DD). Required for query-date.")
    parser.add_argument('--threshold', type=int, default=5000, 
                        help="Traffic threshold for high-traffic command (default: 5000).")

    args = parser.parse_args()

    # LOGGING REQUIREMENT: Log the command and arguments
    logging.info(f"Command invoked: {args.command} | Arguments: {vars(args)}")

    # Load the data
    try:
        df = pd.read_csv(CSV_PATH)
        df['date_time'] = pd.to_datetime(df['date_time'], errors='coerce')
        df = df.dropna(subset=['date_time'])
    except FileNotFoundError:
        logging.error("Data file not found.")
        print("ERROR: Could not find the dataset.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        print("ERROR: Could not process the data.")
        sys.exit(1)

    # ==========================================
    # Execute Commands
    # ==========================================
    if args.command == 'query-date':
        if not args.date:
            logging.error("Missing --date argument for query-date command.")
            print("ERROR: Please provide a date using --date YYYY-MM-DD")
            return
        
        try:
            # Validate date format
            pd.to_datetime(args.date, format='%Y-%m-%d')
        except ValueError:
            logging.error(f"Malformed date provided: {args.date}")
            print("ERROR: Malformed date. Please use the format YYYY-MM-DD.")
            return

        # Filter data
        day_data = df[df['date_time'].dt.date == pd.to_datetime(args.date).date()]
        if day_data.empty:
            print(f"No data found for {args.date}.")
        else:
            print(f"--- Traffic for {args.date} ---")
            print(f"Average Traffic: {day_data['traffic_volume'].mean():.2f} vehicles/hour")
            print(f"Max Traffic: {day_data['traffic_volume'].max()} vehicles/hour")

    elif args.command == 'high-traffic':
        high_traffic_df = df[df['traffic_volume'] > args.threshold]
        print(f"--- High Traffic Periods (>{args.threshold} vehicles/hour) ---")
        print(f"Total hours exceeding threshold: {len(high_traffic_df)}")
        if not high_traffic_df.empty:
            print(f"Most common weather during high traffic: {high_traffic_df['weather_main'].mode()[0]}")

    elif args.command == 'compare-weekend':
        df['is_weekend'] = df['date_time'].dt.dayofweek >= 5
        avg_traffic = df.groupby('is_weekend')['traffic_volume'].mean()
        
        print("--- Weekday vs Weekend Traffic ---")
        print(f"Weekday Average: {avg_traffic.get(False, 0):.2f} vehicles/hour")
        print(f"Weekend Average: {avg_traffic.get(True, 0):.2f} vehicles/hour")
        
        if avg_traffic.get(False, 0) > avg_traffic.get(True, 0):
            print("Conclusion: Weekdays have significantly higher traffic.")
        else:
            print("Conclusion: Weekends have higher traffic.")

if __name__ == "__main__":
    main()
