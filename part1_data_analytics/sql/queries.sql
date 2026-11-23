-- Task 1: Annual Traffic Trends
SELECT SUBSTR(date_time, 1, 4) AS year, SUM(traffic_volume) AS total_volume FROM traffic_data GROUP BY year;
SELECT SUBSTR(date_time, 1, 4) AS year, SUM(traffic_volume) AS total_volume FROM traffic_data GROUP BY year ORDER BY year;

-- Task 2: Descriptive Statistics
SELECT AVG(traffic_volume) AS mean, MIN(traffic_volume) AS min, MAX(traffic_volume) AS max FROM traffic_data;
SELECT AVG(traffic_volume) AS median FROM (SELECT traffic_volume FROM traffic_data ORDER BY traffic_volume LIMIT 2 - (SELECT COUNT(*) FROM traffic_data) % 2 OFFSET (SELECT (COUNT(*) - 1) / 2 FROM traffic_data));

-- Task 3: Probability & Odds Ratio
WITH Counts AS (SELECT COUNT(*) AS total, SUM(CASE WHEN traffic_volume > 5500 THEN 1 ELSE 0 END) AS cong, SUM(CASE WHEN weather_main = 'Clear' THEN 1 ELSE 0 END) AS clear_weather FROM traffic_data) SELECT ROUND(1.0 * cong / total, 4) AS P_Congestion, ROUND(1.0 * clear_weather / total, 4) AS P_Clear FROM Counts;
