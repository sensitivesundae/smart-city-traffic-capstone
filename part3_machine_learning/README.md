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

Task 7: Responsible and Sustainable AI

7.1 Bias and Fairness Report

Sampling and Coverage Limitations: The dataset represents a single traffic corridor (the Metro Interstate). This introduces a severe coverage limitation, as the model's findings may not generalise to urban city centers, rural roads, or regions with different driving cultures and infrastructure. Furthermore, the data lacks actual accident records.

Proxy Labels and Associated Risks: Because no accident data was provided, we used a proxy label (high_risk) defined as severe congestion occurring alongside severe weather. The risk here is label bias: severe congestion does not perfectly equal a high accident risk. A traffic jam caused by a concert is not an accident risk. If Finorg Bank (or a city planner) uses this model to issue safety alerts, it may cause alarm fatigue by flagging safe but congested days, while missing dangerous days that happen to have low traffic volume.

Uneven Error Distribution: Models generally perform worse on rare events (the minority class). In our dataset, extreme weather (like severe snowstorms) or major holidays are rare. Therefore, the prediction error (MAE) will be significantly higher during these extreme conditions. The model is least reliable exactly when the public needs it most (e.g., during a blizzard). Additionally, our regression models performed poorly overall, meaning they struggle to capture the high variance of traffic.

7.2 Governance, Sustainability, and Oversight

Oversight Before Trust: Before this model is trusted for real-world decision-making, a Human-in-the-Loop (HITL) framework must be established. The model should be deployed in "shadow mode" for several months, where its predictions are compared to human traffic controllers' decisions without actually affecting the public. The city must appoint a data governance board to regularly review the model's fairness (e.g., ensuring it doesn't disproportionately flag low-income neighborhoods) and monitor for data drift (as simulated in Task 6).

Environmental and Resource Trade-offs: There is a clear trade-off between model complexity and environmental impact. Training the LSTM (Deep Learning) in Task 3 required much more computational power, time, and energy (higher carbon footprint) compared to the Logistic Regression or Random Forest models. However, the LSTM provided a significantly lower error rate (MAE 612 vs 1800). To balance this, we can use model quantization (an advanced technique from Task 4) to compress the neural network before deployment, reducing the server energy required for real-time predictions. Ultimately, the energy cost of the AI is justified if it successfully reduces traffic idling, which lowers overall city carbon emissions.
