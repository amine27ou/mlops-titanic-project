"""
Airflow DAG for Titanic ML Pipeline (Docker version).
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Docker paths
PROJECT_ROOT = Path("/opt/airflow")
sys.path.insert(0, str(PROJECT_ROOT / "src"))


default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2026, 5, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def validate_data(**context):
    """Validate raw data."""
    import pandas as pd

    print("=" * 60)
    print("TASK: DATA VALIDATION")
    print("=" * 60)

    data_path = PROJECT_ROOT / "data/raw/train.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"✅ Loaded data: {df.shape}")

    assert len(df) == 891, f"Expected 891 rows, got {len(df)}"
    print("✅ Row count correct")

    assert df["PassengerId"].duplicated().sum() == 0, "Duplicate PassengerIds found"
    print("✅ No duplicates")

    print("✅ VALIDATION PASSED")

    context["ti"].xcom_push(key="row_count", value=len(df))
    context["ti"].xcom_push(key="survival_rate", value=float(df["Survived"].mean()))


def preprocess_data(**context):
    """Preprocess data."""
    from data.load_data import load_raw_data
    from data.preprocessing import preprocess_data as prep, split_data

    print("=" * 60)
    print("TASK: PREPROCESSING")
    print("=" * 60)

    df = load_raw_data(str(PROJECT_ROOT / "data/raw/train.csv"))
    X, y = prep(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")

    context["ti"].xcom_push(key="train_size", value=len(X_train))
    context["ti"].xcom_push(key="test_size", value=len(X_test))


def train_simple_model(**context):
    """Train a simple model (lightweight for Docker)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from data.load_data import load_raw_data
    from data.preprocessing import preprocess_data as prep, split_data
    import joblib

    print("=" * 60)
    print("TASK: TRAINING")
    print("=" * 60)

    df = load_raw_data(str(PROJECT_ROOT / "data/raw/train.csv"))
    X, y = prep(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ F1 Score: {f1:.4f}")

    # Save model
    model_path = PROJECT_ROOT / "models/pipeline_model.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved: {model_path}")

    context["ti"].xcom_push(key="accuracy", value=float(accuracy))
    context["ti"].xcom_push(key="f1_score", value=float(f1))


def generate_report(**context):
    """Generate pipeline report."""
    import pandas as pd

    print("=" * 60)
    print("TASK: REPORTING")
    print("=" * 60)

    ti = context["ti"]

    report = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_rows": ti.xcom_pull(task_ids="validate_data", key="row_count"),
        "survival_rate": ti.xcom_pull(task_ids="validate_data", key="survival_rate"),
        "train_size": ti.xcom_pull(task_ids="preprocess_data", key="train_size"),
        "test_size": ti.xcom_pull(task_ids="preprocess_data", key="test_size"),
        "accuracy": ti.xcom_pull(task_ids="train_model", key="accuracy"),
        "f1_score": ti.xcom_pull(task_ids="train_model", key="f1_score"),
        "status": "SUCCESS",
    }

    print("\nPipeline Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    # Save report
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    report_df = pd.DataFrame([report])
    report_path = reports_dir / "pipeline_runs.csv"

    if report_path.exists():
        existing = pd.read_csv(report_path)
        report_df = pd.concat([existing, report_df], ignore_index=True)

    report_df.to_csv(report_path, index=False)
    print(f"\n✅ Report saved: {report_path}")


with DAG(
    "titanic_ml_pipeline",
    default_args=default_args,
    description="Titanic ML Pipeline",
    schedule_interval="@daily",
    catchup=False,
    tags=["ml", "titanic"],
) as dag:
    t1 = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_simple_model,
    )

    t4 = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
    )

    t5 = BashOperator(
        task_id="complete",
        bash_command='echo "Pipeline completed successfully"',
    )

    t1 >> t2 >> t3 >> t4 >> t5
