### Feature Engineering Logic
*   **Cyclical Encoding:** `hour_sin` and `hour_cos` are used to capture the circular nature of time (so hour 23 and hour 0 are mathematically close).
*   **Weather Encoding:** One-Hot Encoding is applied to `weather_main` to make it ML-ready. `is_raining` and `is_snowing` are boolean flags created from rainfall/snowfall columns.
*   **Normalisation:** Min-Max scaling is applied to `temp` and `clouds_all` so values are bounded between 0 and 1.
*   **Target Definition:** The `traffic_category` target is created as follows:
    *   **Low:** < 4,500 vehicles/hour
    *   **Medium:** 4,500 - 5,500 vehicles/hour
    *   **High:** > 5,500 vehicles/hour
