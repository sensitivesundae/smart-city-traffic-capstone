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
1. **Traffic demand by hour:** Traffic volume follows a bimodal (two-peak) pattern, sharply rising around 7-8 AM and again at 4-5 PM, which perfectly aligns with standard morning and evening commuting hours. Volume drops to its lowest between midnight and 4 AM.
2. **Weekday vs. weekend traffic:** Weekdays have a significantly higher average traffic volume than weekends. This confirms that the primary driver of traffic in this dataset is work-related commuting rather than leisure travel.
3. **Temperature vs. traffic:** There is no strong linear relationship. However, the highest traffic volumes cluster around mild temperatures (10°C to 30°C). Extreme cold (below -10°C) tends to have lower traffic, likely due to travel disruptions or people staying home.
