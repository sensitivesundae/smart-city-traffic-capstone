## Part 3: Machine Learning & Intelligent Mobility Solution

### Dataset & Proxy Disclaimer
**No real accident dataset was provided for this capstone.** 
As instructed in the assignment prompt, a **documented proxy label** was created to enable the classification task. A record was classified as `high_risk` (Proxy Accident Risk) when severe/high congestion occurred together with severe or low-visibility weather conditions. This proxy is intended only to demonstrate the ML workflow and should not be interpreted as a prediction of actual accidents.

### Project Structure
*   `task1_supervised.py`: Trains Logistic Regression, Random Forest (Classification), and Linear Regression, Random Forest (Regression).
*   `task2_unsupervised.py`: Implements K-Means clustering and Association Rule Mining (Apriori).
*   `task3_deep_learning.py`: Implements an LSTM neural network and generates a SHAP explainability plot.
*   `task4_mlops.py`: Demonstrates MLflow experiment tracking.
*   `task5_recommendation.py`: A CLI recommendation engine for travel times.
*   `task6_api_deployment.py`: A FastAPI mock deployment for real-time prediction.
*   `task6_monitoring.py`: A monitoring and alerting simulation for model drift.
*   `FINAL_REPORT.md`: The written methodology, findings, and bias/fairness report.

### How to Run
Ensure you have installed: `pip install pandas numpy scikit-learn mlxtend tensorflow shap mlflow fastapi uvicorn`
Run any script via terminal, for example: `python task1_supervised.py`
