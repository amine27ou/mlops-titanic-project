"""FastAPI application for Titanic survival prediction."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="MLOps project - model serving endpoint",
    version="1.0.0",
)

MODEL_PATH = os.getenv("MODEL_PATH", "/app/model/model.pkl")
model = None


def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same preprocessing as training."""
    df = df.copy()

    # Fill missing Age (use a safe constant or training median if you know it)
    df["Age"] = df["Age"].fillna(28)

    # Encode Sex: male=1, female=0
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # Ensure required columns exist (safety)
    expected_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    # Return only what the model expects
    return df[expected_cols]


@app.on_event("startup")
async def load_model():
    """Load model from pickle file."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise


class PassengerData(BaseModel):
    """Input schema for prediction."""

    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Sex: str = Field(..., description="Gender (male or female)")
    Age: float = Field(..., ge=0, le=100, description="Age in years")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Ticket fare")
    Embarked: str = Field(..., description="Port of embarkation (C, Q, or S)")

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
    """Output schema for prediction."""

    survived: int
    probability: float
    passenger_class: int


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": "Titanic Survival Predictor",
        "version": "1.0.0",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerData):
    """
    Predict survival probability for a Titanic passenger.

    Returns:
        survived: 0 (died) or 1 (survived)
        probability: Confidence score [0-1]
        passenger_class: Echo of input class
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([passenger.dict()])

        # Preprocess
        processed_df = preprocess_input(input_df)

        # Get prediction
        prediction = model.predict(processed_df)[0]

        # Get probability
        try:
            proba = model.predict_proba(processed_df)[0][1]
        except Exception as e:
            proba = float(prediction)
            print(e)

        return PredictionResponse(
            survived=int(prediction),
            probability=float(proba),
            passenger_class=passenger.Pclass,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(passengers: list[PassengerData]):
    """Batch prediction endpoint."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([p.dict() for p in passengers])
        processed_df = preprocess_input(input_df)
        predictions = model.predict(processed_df)

        return {"predictions": predictions.tolist(), "count": len(predictions)}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}"
        )
