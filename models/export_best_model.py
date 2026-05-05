import importlib.util
from pathlib import Path
import mlflow
import joblib
import os

# Import preprocessing module directly
preprocessing_path = Path(__file__).parent.parent / "src" / "data" / "preprocessing.py"
spec = importlib.util.spec_from_file_location("preprocessing", preprocessing_path)
preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocessing)
preprocess_data = preprocessing.preprocess_data


def export_best_model():
    """Find best model and export as pickle."""
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")

    # Get best run
    client = mlflow.tracking.MlflowClient()

    runs = client.search_runs(
        experiment_ids=["0"], order_by=["metrics.accuracy DESC"], max_results=1
    )

    if not runs:
        print("❌ No runs found")
        return

    best_run = runs[0]
    run_id = best_run.info.run_id

    print(f"✅ Best run: {run_id}")
    print(f"   Accuracy: {best_run.data.metrics['accuracy']:.4f}")

    # Load model
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)

    # Create export directory
    os.makedirs("api/model", exist_ok=True)

    # Save model
    joblib.dump(model, "api/model/model.pkl")

    # Save metadata
    with open("api/model/metadata.txt", "w") as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Accuracy: {best_run.data.metrics['accuracy']:.4f}\n")
        f.write(f"Model: {best_run.data.params.get('model_type', 'unknown')}\n")

    print("✅ Model exported to api/model/model.pkl")


if __name__ == "__main__":
    export_best_model()
