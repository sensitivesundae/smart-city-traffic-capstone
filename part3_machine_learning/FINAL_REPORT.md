
# Part 3 Final Capstone Report: Intelligent Mobility Solution

## Dataset & Proxy Disclaimer
**No real accident dataset was provided for this capstone.** 
As instructed in the assignment prompt, a documented proxy label was created to enable the classification task. A record was classified as `high_risk` (Proxy Accident Risk) when severe/high congestion occurred together with severe or low-visibility weather conditions. This proxy is intended only to demonstrate the ML workflow and should not be interpreted as a prediction of actual accidents.

---

## Task 1: Supervised Machine Learning Models
**Methodology:**
The goal was to predict both a discrete label (High Risk) and a continuous value (Traffic Volume). We engineered time-based features (Hour, Day of Week), cyclical encodings (`hour_sin`, `hour_cos`), weather indicators (`is_raining`), and holiday flags. 
*   **Classification:** We trained a Logistic Regression baseline and a Random Forest Classifier to predict the `high_risk` proxy label.
*   **Regression:** We trained a Linear Regression baseline and a Random Forest Regressor to predict exact `traffic_volume`.

**Findings:**
*   **Classification:** The Random Forest Classifier significantly outperformed the baseline, achieving an **Accuracy of 98.02%**, an **F1-Score of 94.89%**, and a **ROC AUC of 0.9966**. The tree-based model successfully captured non-linear relationships (e.g., traffic is only severe when it is *both* raining and rush hour).
*   **Regression:** Both regression models struggled, yielding negative R-squared values (-0.0014 for Linear, -0.1218 for Random Forest). Predicting exact traffic volume using only static weather/time features is highly inaccurate because traffic is heavily dependent on sequential historical momentum (e.g., what happened in the previous hour).

---

## Task 2: Unsupervised Machine Learning
**Methodology:**
We applied two unsupervised techniques:
1.  **K-Means Clustering:** Using `hour`, `weather_severity`, and `traffic_volume`, we grouped the data into 4 clusters.
2.  **Association Rule Mining (Apriori):** We discretized time into categories (Morning Rush, Midday, etc.), weather (Clear, Severe), and Congestion levels to find rules predicting severe congestion.

**Findings:**
*   **Clustering:** The algorithm identified 4 distinct profiles: Nighttime Lows (3:00 AM, ~872 vehicles), Midday Clear (12:30 PM, ~5,045 vehicles), Midday Bad Weather (12:30 PM, ~4,601 vehicles), and Late Evening (8:30 PM, ~2,521 vehicles). Time of day was the dominant clustering factor.
*   **Association Rules:** The rule with the highest Lift (3.338) was: `IF {Morning_Rush, Weekday, Clear} THEN {Severe Congestion}`. The high lift indicates that this specific combination of time and day makes severe traffic 3.3 times more likely than random chance. Weather was mostly absent from the top rules, proving human commuting schedules dictate congestion far more than weather.

---

## Task 3: Deep Learning with Explainability
**Methodology:**
*   **Deep Learning:** An LSTM (Long Short-Term Memory) neural network was implemented to predict traffic demand. It was fed sequences of the past 24 hours of data to capture temporal dependencies.
*   **Explainability:** Because LSTMs operate on 3D sequential data and are difficult to explain directly, a Proxy Random Forest Regressor was trained on the same tabular features to generate SHAP (SHapley Additive exPlanations) values. SHAP was chosen over LIME for its mathematical consistency and global interpretability.

**Findings:**
*   **LSTM:** The deep learning model achieved a **Mean Absolute Error (MAE) of 612.83 vehicles**, drastically outperforming the Random Forest Regressor from Task 1 (MAE ~1,800). This proves that sequential data is critical for traffic forecasting.
*   **SHAP:** The SHAP summary plot revealed that `hour` is the overwhelmingly dominant feature (mean SHAP value of ~1600), followed by `is_weekend` (~225). Weather features (temp, rain) had near-zero impact, confirming that structural human routines are the primary driver of traffic demand.

---

## Task 4: Advanced AI Technique (MLflow)
**Methodology:**
We selected **MLflow for experiment tracking** as it directly supports the MLOps lifecycle. Two Random Forest Regressor models (50 trees vs. 100 trees) were trained. MLflow was used to log the hyperparameters, evaluation metrics (MAE), and the serialized model artifacts using a local SQLite database (`mlflow.db`).

**Findings:**
MLflow provides critical value by ensuring reproducibility and transparency. Instead of manually recording model scores, MLflow automatically tracks every iteration, allowing data scientists to compare runs in a UI dashboard and select the best model for production. Its limitation is that a local SQLite database does not support team collaboration; a remote tracking server would be required for enterprise use.

---

## Task 5: Traffic Recommendation System
**Methodology:**
The system translates historical data into actionable advice, focusing on **travel timing** since the dataset represents a single corridor. The system accepts user inputs (Weekday/Weekend and Weather Condition), filters the dataset, and groups it by hour to calculate the average traffic volume. It identifies the 2-hour window with the lowest historical traffic and generates a plain-language recommendation.

**Findings:**
The system successfully generates dynamic advice. For example: *"For a weekday journey during clear weather, consider travelling between 10:00 AM and 12:00 PM, when historical traffic volumes are typically lower (average of 3,250 vehicles/hour)."* This adds practical business value by helping commuters avoid peak congestion, reducing time lost and carbon emissions.

---

## Task 6: MLOps and Deployment Simulation
**Methodology:**
*   **Deployment:** A mock API was built using **FastAPI**. The script loads the trained Random Forest model and exposes a `/predict` POST endpoint that accepts JSON input and returns a real-time traffic prediction.
*   **Monitoring & Alerting:** A monitoring script was created to simulate checking new incoming data against the baseline training data. It checks for **Prediction Error Drift** (triggering an alert if MAE increases by 20%) and **Feature Distribution Drift** (triggering an alert if the mean temperature shifts by 5 Kelvin).

**Findings:**
The simulated dashboard successfully outputs a `🟢 STATUS: PASS / Normal` when data is stable, and a `🔴 STATUS: ALERT / Requires Investigation` if drift is detected. This simulates a real-world MLOps pipeline where automated alerts notify engineers of model degradation before it impacts users.

---

## Task 7: Responsible and Sustainable AI

### 7.1 Bias and Fairness Report
*   **Sampling Limitations:** The dataset represents a single traffic corridor (Metro Interstate). This introduces a severe coverage limitation, as the model's findings may not generalise to urban city centers or regions with different driving cultures.
*   **Proxy Labels:** The proxy label (`high_risk` = severe congestion + severe weather) carries the risk of **label bias**. A traffic jam caused by a concert is not an accident risk. If a city uses this model to issue safety alerts, it may cause alarm fatigue by flagging safe but congested days, while missing dangerous days with low traffic.
*   **Uneven Error Distribution:** Models generally perform worse on rare events (minority classes). Extreme weather (snowstorms) and holidays are rare. Therefore, prediction errors will be highest during extreme conditions—exactly when the public needs the system most.

### 7.2 Governance, Sustainability, and Oversight
*   **Oversight Before Trust:** A **Human-in-the-Loop (HITL)** framework must be established. The model should be deployed in "shadow mode" for several months, comparing its predictions to human traffic controllers without affecting the public. A governance board must regularly audit the model for fairness and data drift.
*   **Resource Trade-offs:** There is a trade-off between complexity and environmental impact. Training the LSTM (Task 3) required significantly more compute and energy than Logistic Regression. However, the LSTM was far more accurate (MAE 612 vs 1800). To balance this, **model quantization** should be applied before deployment to compress the neural network, reducing server energy consumption while maintaining accuracy. The energy cost of the AI is justified if it successfully reduces city-wide traffic idling, lowering overall carbon emissions.
