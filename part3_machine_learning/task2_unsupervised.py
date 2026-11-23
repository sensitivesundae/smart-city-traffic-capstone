import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mlxtend.frequent_patterns import apriori, association_rules
import warnings
warnings.filterwarnings('ignore')

# 1. Load and Prep Data
print("Loading data...")
file_path = r'C:\Users\Lina\OneDrive\nus ai\capstone\Metro_Interstate_Traffic_Volume.csv'
df = pd.read_csv(file_path)
df['date_time'] = pd.to_datetime(df['date_time'])
df['weather_main'] = df['weather_main'].str.lower().str.strip()

# ==========================================
# TASK 2.1: K-Means Clustering
# ==========================================
print("\n--- K-Means Clustering ---")

# Create features for clustering
df['hour'] = df['date_time'].dt.hour
df['is_weekend'] = (df['date_time'].dt.dayofweek >= 5).astype(int)

# Create a simple weather severity score (0 = clear, 1 = clouds, 2 = rain/snow/fog)
def weather_severity(w):
    if w in ['clear', 'clouds']: return 0
    elif w in ['rain', 'drizzle', 'mist', 'haze', 'smoke']: return 1
    else: return 2 # Snow, Thunderstorm, Squall, Fog
df['weather_severity'] = df['weather_main'].apply(weather_severity)

# Select features for K-Means
cluster_features = ['hour', 'weather_severity', 'traffic_volume']
X_cluster = df[cluster_features]

# Scale the features (crucial for K-Means so traffic_volume doesn't overpower hour)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Train K-Means (using 4 clusters for interpretability)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Show cluster centers (unscale them to make them readable)
centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_summary = pd.DataFrame(centers, columns=cluster_features)
cluster_summary['count'] = df['cluster'].value_counts().sort_index().values

print("\nK-Means Cluster Centers (Interpretation):")
print(cluster_summary.round(1).to_string())


# ==========================================
# TASK 2.2: Association Rule Mining
# ==========================================
print("\n--- Association Rule Mining ---")

# 1. Discretise variables
def time_of_day(h):
    if 6 <= h < 10: return 'Morning_Rush'
    elif 10 <= h < 15: return 'Midday'
    elif 15 <= h < 19: return 'Evening_Rush'
    else: return 'Night'

df['TimeCat'] = df['hour'].apply(time_of_day)
df['WeekendCat'] = df['is_weekend'].map({0: 'Weekday', 1: 'Weekend'})
df['WeatherCat'] = df['weather_severity'].map({0: 'Clear', 1: 'Mild_Weather', 2: 'Severe_Weather'})

# Congestion level (using quantiles)
q1, q2, q3 = df['traffic_volume'].quantile([0.25, 0.5, 0.75]).values
def cong_bucket(v):
    if v <= q1: return 'Congestion_Low'
    elif v <= q2: return 'Congestion_Med'
    elif v <= q3: return 'Congestion_High'
    return 'Congestion_Severe'
df['Congestion'] = df['traffic_volume'].apply(cong_bucket)

# 2. One-hot encode the categorical variables
basket = pd.get_dummies(df[['TimeCat', 'WeekendCat', 'WeatherCat', 'Congestion']])

# 3. Apply Apriori algorithm (min_support=0.05 means rule must appear in at least 5% of data)
frequent_itemsets = apriori(basket, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# 4. Filter for rules that predict Congestion_Severe or High, and sort by Lift
congestion_rules = rules[
    rules['consequents'].astype(str).str.contains('Congestion_High|Congestion_Severe')
]
top_rules = congestion_rules.sort_values(by='lift', ascending=False).head(5)

print("\nTop 5 Association Rules Predicting Congestion (Sorted by Lift):")
for index, row in top_rules.iterrows():
    antecedents = list(row['antecedents'])
    consequents = list(row['consequents'])
    print(f"IF {antecedents} THEN {consequents}")
    print(f"   Support: {row['support']:.3f}, Confidence: {row['confidence']:.3f}, Lift: {row['lift']:.3f}\n")
