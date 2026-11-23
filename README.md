## Task 5: Reproducibility & Project Structure

### Project Structure
*   `part2_python/`
    *   `pipeline.py`: Main entry point for data loading, cleaning, and orchestration.
    *   `feature_engineering.py`: Module containing functions to create ML features (time, weather, normalisation).
    *   `visualizations.py`: Module containing functions to generate Matplotlib/Seaborn charts.
    *   `run_visualizations.py`: Entry point script to run the visualization module.
    *   `mini_app/`: Contains the command-line application (`app.py`) for querying the dataset.
    *   `figures/`: Directory where generated `.png` charts are saved.
    *   `pipeline.log`, `visualizations.log`, `mini_app/app.log`: Log files.

### How to Run
1.  Ensure you have the required libraries: `pip install pandas numpy matplotlib seaborn`
2.  Update the `file_path` variable in `pipeline.py` and `run_visualizations.py` to point to your local CSV file.
3.  Run the data pipeline: `python pipeline.py`
4.  Generate visualizations: `python run_visualizations.py`
5.  Run the CLI app (example): `python mini_app/app.py compare-weekend`

### Logging Configuration
We use Python's built-in `logging` module to track system events.
*   **Where logs are written:**
    *   `pipeline.log`: Records data loading, cleaning, and feature engineering.
    *   `visualizations.log`: Records chart generation and file saving.
    *   `mini_app/app.log`: Records CLI commands and user input errors.
    *   *All logs are also printed to the console for real-time monitoring (except the CLI app, which prints outputs directly to the user).*
*   **Log Level Meanings:**
    *   `DEBUG`: Fine-grained internal values (e.g., min/max values calculated during normalisation).
    *   `INFO`: Normal milestones (e.g., data loaded, step completed, file saved).
    *   `WARNING`: Recoverable issues (e.g., duplicate rows dropped, missing values imputed).
    *   `ERROR`: A failure that prevents the pipeline from continuing (e.g., file not found, malformed user input).


## Feature Engineering Logic & Target Definition

### 1. Time & Weather Features
*   **Cyclical Encoding:** `hour_sin` and `hour_cos` are applied to the 'Hour' feature to capture the circular nature of time (so hour 23 is mathematically close to hour 0).
*   **Weather Encoding:** One-Hot Encoding (`pd.get_dummies`) is applied to `weather_main`. Additionally, binary indicators `is_raining` and `is_snowing` are created based on whether the `rain_1h` or `snow_1h` columns are greater than 0.

### 2. Normalisation
*   Min-Max scaling is applied to `temp` and `clouds_all` to bring their values between 0 and 1, making them ML-ready.

### 3. Traffic Target Definition
The target variable `traffic_category` is created using the following data-driven logic based on hourly vehicle volume:
*   **Low:** < 4,500 vehicles per hour
*   **Medium:** 4,500 to 5,500 vehicles per hour
*   **High:** > 5,500 vehicles per hour

## Task 3: Visualisation Interpretations
<img width="1000" height="500" alt="traffic_by_hour" src="https://github.com/user-attachments/assets/5a4091d0-08f2-47cc-ae06-14d35660526b" />

1. **Traffic demand by hour:** Traffic volume follows a bimodal (two-peak) pattern, sharply rising around 7-8 AM and again at 4-5 PM, which perfectly aligns with standard morning and evening commuting hours. Volume drops to its lowest between midnight and 4 AM.

<img width="800" height="500" alt="weekday_vs_weekend" src="https://github.com/user-attachments/assets/f357c602-2e5b-4c27-a45d-4179a45e01f5" />

2. **Weekday vs. weekend traffic:** Weekdays have a significantly higher average traffic volume than weekends. This confirms that the primary driver of traffic in this dataset is work-related commuting rather than leisure travel.

<img width="1000" height="500" alt="weather_impact" src="https://github.com/user-attachments/assets/fadcaa46-294f-4a2d-948b-82fa31f49477" />

3. **Weather vs. traffic:** There is no strong linear relationship. However, the highest traffic volumes cluster around mild weather (cloudy, haze). Extreme weather (squall and fog) tends to have lower traffic, likely due to travel disruptions or people staying home.
