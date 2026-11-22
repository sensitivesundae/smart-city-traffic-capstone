import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(df):
    """
    Creates and saves the required visualizations.
    """
    # Set the style
    sns.set_theme(style="whitegrid")
    
    # Make sure date_time is a datetime object
    df['date_time'] = pd.to_datetime(df['date_time'])

    # ==========================================
    # A. Daily traffic trends (2015, 2016, 2017)
    # ==========================================
    # Filter for 2015 to 2017
    daily_df = df[(df['date_time'].dt.year >= 2015) & (df['date_time'].dt.year <= 2017)]
    
    plt.figure(figsize=(12, 6))
    # Group by date and calculate average traffic volume
    daily_trend = daily_df.groupby(daily_df['date_time'].dt.date)['traffic_volume'].mean()
    plt.plot(daily_trend.index, daily_trend.values, color='blue', alpha=0.6)
    
    plt.title('Daily Average Traffic Trends (2015-2017)')
    plt.xlabel('Date')
    plt.ylabel('Average Traffic Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/daily_traffic_trends.png')
    plt.show()

    # ==========================================
    # B. Hourly traffic patterns (2017)
    # ==========================================
    hourly_df = df[df['date_time'].dt.year == 2017]
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=hourly_df, x='hour', y='traffic_volume', estimator='mean', color='teal')
    
    plt.title('Average Hourly Traffic Volume (2017)')
    plt.xlabel('Hour of the Day (0-23)')
    plt.ylabel('Average Traffic Volume')
    plt.tight_layout()
    plt.savefig('figures/hourly_traffic_patterns.png')
    plt.show()

    # ==========================================
    # C. Weather impact
    # ==========================================
    plt.figure(figsize=(10, 5))
    # Sort weather by average traffic
    weather_order = df.groupby('weather_main')['traffic_volume'].mean().sort_values(ascending=False).index
    sns.barplot(data=df, x='weather_main', y='traffic_volume', order=weather_order, estimator='mean', color='coral')
    
    plt.title('Average Traffic Volume by Weather Condition')
    plt.xlabel('Weather Condition')
    plt.ylabel('Average Traffic Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('figures/weather_impact.png')
    plt.show()

if __name__ == "__main__":
    print("Visualization module loaded. Use create_visualizations(df) to generate plots.")
