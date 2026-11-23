Task 6: MLOps and Deployment Simulation

6.1 Model Versioning & 6.2 Experiment Tracking
Model versioning is managed using MLflow (as demonstrated in Task 4). By executing mlflow.start_run(), every model iteration is assigned a unique Run ID. MLflow tracks the hyperparameters (e.g., n_estimators=50 vs 100), the evaluation metrics (MAE), and serializes the actual model artifacts. This allows Finorg Bank to compare models, ensure reproducibility, and easily roll back to a previous version if a newer model fails in production.

6.3 Deployment
A mock deployment was created using FastAPI (task6_api_deployment.py). The script initializes a Random Forest model and exposes a /predict POST endpoint. The endpoint accepts a JSON payload containing features (hour, day of week, temp, rain, clouds) and returns the predicted traffic volume in real-time.

6.4 Monitoring & 6.5 Alerting
A monitoring dashboard simulation (task6_monitoring.py) was created to continuously evaluate the deployed model.

Prediction Error Drift: The script evaluates the model on new incoming data. If the Mean Absolute Error (MAE) exceeds a 20% threshold compared to the baseline, an alert is triggered.

Feature Distribution Drift: The script compares the mean temperature of the training data against the incoming data. If the temperature distribution shifts by more than 5 Kelvin, it flags a data drift.

Alerting Mechanism: If either drift threshold is breached, the system prints 🔴 STATUS: ALERT / Requires Investigation. Otherwise, it outputs 🟢 STATUS: PASS / Normal.
