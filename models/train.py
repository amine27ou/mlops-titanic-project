"""
Model training script for Titanic survival prediction.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    classification_report,
    confusion_matrix
)
import joblib
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.load_data import load_raw_data
from data.preprocessing import preprocess_data, split_data


def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model."""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """Train Random Forest model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str):
    """
    Evaluate model and print metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model for display
        
    Returns:
        Dictionary with metrics
    """
    y_pred = model.predict(X_test)
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} RESULTS")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    
    return metrics


def save_model(model, model_name: str, output_dir: str = "models"):
    """Save trained model to disk."""
    Path(output_dir).mkdir(exist_ok=True)
    filepath = f"{output_dir}/{model_name}.pkl"
    joblib.dump(model, filepath)
    print(f"\n✅ Model saved to {filepath}")


def main():
    """Main training pipeline."""
    print("="*60)
    print("TITANIC SURVIVAL PREDICTION - BASELINE MODELS")
    print("="*60)
    
    # Load and preprocess data
    df = load_raw_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    print("\nTraining Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Compare models
    results_df = pd.DataFrame([lr_metrics, rf_metrics])
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # Save results
    Path("reports").mkdir(exist_ok=True)
    results_df.to_csv("reports/baseline_results.csv", index=False)
    print("\n✅ Results saved to reports/baseline_results.csv")
    
    # Save best model
    best_model_name = results_df.loc[results_df['accuracy'].idxmax(), 'model']
    best_model = rf_model if best_model_name == "Random Forest" else lr_model
    save_model(best_model, "best_model")
    
    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE - Best model: {best_model_name}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()  