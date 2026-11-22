import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# Set up logger
logger = logging.getLogger(__name__)

def create_visualizations(df):
    """
    Creates and saves the required visualizations.
    """
    # Ensure the figures directory exists
    if not os.path.exists('figures'):
        os.makedirs('figures')

    sns.set_theme(style="whitegrid")
    df['date_time'] = pd.to_datetime(df['date_time'])

    # 1. Traffic demand by hour
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x=df['date_time'].dt.hour, y='traffic_volume', estimator='mean', color='teal')
    plt.title('Average Traffic Volume by Hour of Day')
    plt.xlabel('Hour of the Day (0-23)')
    plt.ylabel('Average Traffic Volume')
    plt.tight_layout()
    file_path_1 = 'figures/traffic_by_hour.png'
    plt.savefig(file_path_1)
    logger.info(f"Figure successfully saved: {file_path_1}") # LOGGING REQUIREMENT
    plt.show()

    # 2. Weekday vs. Weekend Traffic
    plt.figure(figsize=(8, 5))
    df['is_weekend'] = df['date_time'].dt.dayofweek >= 5
    sns.barplot(data=df, x='is_weekend', y='traffic_volume', estimator='mean', palette=['blue', 'orange'])
    plt.title('Average Traffic: Weekday vs Weekend')
    plt.xlabel('Is Weekend? (False = Weekday, True = Weekend)')
    plt.ylabel('Average Traffic Volume')
    plt.tight_layout()
    file_path_2 = 'figures/weekday_vs_weekend.png'
    plt.savefig(file_path_2)
    logger.info(f"Figure successfully saved: {file_path_2}") # LOGGING REQUIREMENT
    plt.show()

    # 3. Temperature vs. Traffic
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=df, x=df['temp'] - 273.15, y='traffic_volume', alpha=0.3, color='purple')
    plt.title('Temperature vs. Traffic Volume')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Traffic Volume')
    plt.tight_layout()
    file_path_3 = 'figures/temp_vs_traffic.png'
    plt.savefig(file_path_3)
    logger.info(f"Figure successfully saved: {file_path_3}") # LOGGING REQUIREMENT
    plt.show()

if __name__ == "__main__":
    print("Visualization module loaded.")
