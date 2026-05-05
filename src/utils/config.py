"""
Configuration settings for the project.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# MLflow settings
MLFLOW_TRACKING_URI = "file:./mlruns"
MLFLOW_EXPERIMENT_NAME = "titanic-survival-prediction"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature configuration
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
TARGET = "Survived"
