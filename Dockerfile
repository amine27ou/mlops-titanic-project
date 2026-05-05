FROM apache/airflow:2.7.3-python3.10

USER root
RUN pip install --no-cache-dir pandas numpy scikit-learn xgboost mlflow joblib

USER airflow
