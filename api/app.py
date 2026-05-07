import os
import time
import logging
import numpy as np
import pandas as pd
import joblib
import mlflow  # Added
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response


# ====================== LOGGING ======================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="MLOps project - model serving endpoint",
    version="1.0.0",
)

# ====================== PROMETHEUS METRICS ======================
prediction_counter = Counter(
    "titanic_predictions_total",
    "Total number of predictions made",
    ["survival_outcome"],
)

prediction_duration = Histogram(
    "titanic_prediction_duration_seconds",
    "Time spent processing prediction",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

model_loaded_gauge = Gauge(
    "titanic_model_loaded", "Whether the model is loaded (1 = yes, 0 = no)"
)

# ====================== MODEL LOADING ======================
MODEL_PATH = os.getenv("MODEL_PATH", "./api/model/model.pkl")
model = None


@app.on_event("startup")
async def load_model():
    """Load model from pickle file on startup."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        model_loaded_gauge.set(1)
        logger.info(f"✅ Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        model_loaded_gauge.set(0)
        logger.error(f"❌ Failed to load model from {MODEL_PATH}: {e}")
        raise


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as training pipeline."""
    df = df.copy()

    # Fill missing values (match your training preprocessing)
    df["Age"] = df["Age"].fillna(28)  # or use training median
    # df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    # Encode Sex
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0}).fillna(0)

    # Select only features the model was trained on
    expected_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df[expected_cols]


class PassengerData(BaseModel):
    Pclass: int = Field(..., ge=1, le=3)
    Sex: str = Field(..., description="male or female")
    Age: float = Field(..., ge=0, le=100)
    SibSp: int = Field(..., ge=0)
    Parch: int = Field(..., ge=0)
    Fare: float = Field(..., ge=0)
    Embarked: str = Field(..., description="C, Q or S")  # kept for future use

    class Config:
        json_schema_extra = {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "Age": 22.0,
                "SibSp": 1,
                "Parch": 0,
                "Fare": 7.25,
                "Embarked": "S",
            }
        }


class PredictionResponse(BaseModel):
    survived: int
    probability: float
    passenger_class: int


@app.get("/")
def read_root():
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "timestamp": time.time(),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerData):
    start_time = time.time()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([passenger.dict()])
        processed_df = preprocess_input(input_df)

        # Predict
        prediction = model.predict(processed_df)[0]  # scalar
        proba = (
            model.predict_proba(processed_df)[0][1]
            if hasattr(model, "predict_proba")
            else float(prediction)
        )

        outcome_label = "Survived" if int(prediction) == 1 else "Did not survive"

        # ====================== PROMETHEUS ======================
        prediction_counter.labels(survival_outcome=outcome_label).inc()
        duration = time.time() - start_time
        prediction_duration.observe(duration)

        # ====================== MLflow MONITORING (10% sampling) ======================
        if np.random.random() < 0.1:
            try:
                with mlflow.start_run(run_name="api_prediction"):
                    mlflow.log_param("pclass", passenger.Pclass)
                    mlflow.log_param("sex", passenger.Sex)
                    mlflow.log_metric("predicted_survived", int(prediction))
                    mlflow.log_metric("survival_probability", float(proba))
                    mlflow.log_metric("prediction_duration", duration)
            except Exception as e:
                logger.warning(f"MLflow logging failed (non-critical): {e}")

        logger.info(
            f"Prediction → {outcome_label} | Prob: {proba:.3f} | Duration: {duration:.3f}s"
        )

        return PredictionResponse(
            survived=int(prediction),
            probability=float(proba),
            passenger_class=passenger.Pclass,
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(passengers: list[PassengerData]):
    start_time = time.time()

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([p.dict() for p in passengers])
        processed_df = preprocess_input(input_df)

        predictions = model.predict(processed_df)
        probas = (
            model.predict_proba(processed_df)[:, 1]
            if hasattr(model, "predict_proba")
            else predictions.astype(float)
        )

        # Prometheus metrics
        for pred in predictions:
            outcome_label = "Survived" if int(pred) == 1 else "Did not survive"
            prediction_counter.labels(survival_outcome=outcome_label).inc()

        duration = time.time() - start_time
        prediction_duration.observe(duration)

        logger.info(
            f"Batch prediction: {len(predictions)} passengers in {duration:.3f}s"
        )

        return {
            "predictions": predictions.tolist(),
            "probabilities": probas.tolist(),
            "count": len(predictions),
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================== PROMETHEUS METRICS ENDPOINT ======================
@app.get("/metrics", include_in_schema=False)
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Auto-instrumentation
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics_internal", include_in_schema=False
)
