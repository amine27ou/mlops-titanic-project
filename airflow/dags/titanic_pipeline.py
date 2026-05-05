from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 5, 5),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    "titanic_ml_pipeline",
    default_args=default_args,
    description="MLOps Titanic Pipeline",
    schedule_interval=None,  # Manual trigger only
    catchup=False,
    tags=["mlops", "titanic"],
) as dag:
    validate_data = BashOperator(
        task_id="validate_raw_data",
        bash_command="cd /opt/airflow/project && python -m pytest tests/test_load_data.py -v",
    )

    preprocess_data = BashOperator(
        task_id="preprocess_data",
        bash_command="cd /opt/airflow/project && python src/data/create_processed_data.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command="cd /opt/airflow/project && python src/models/train_with_mlflow.py",
    )

    validate_data >> preprocess_data >> train_model
