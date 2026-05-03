"""
Model training with MLflow tracking.
Logs parameters, metrics, and model artifacts.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
import sys
from pathlib import Path

# Add repository root and src directory to path so both data/ and src/utils/ imports resolve
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from data.load_data import load_raw_data
from data.preprocessing import preprocess_data, split_data
from utils.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME


def setup_mlflow():
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    print(f"✅ MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"✅ MLflow experiment: {MLFLOW_EXPERIMENT_NAME}")


def log_metrics(y_test, y_pred, y_pred_proba=None):
    """
    Calculate and log metrics to MLflow.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    # Log to MLflow
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # Log confusion matrix as text
    cm = confusion_matrix(y_test, y_pred)
    mlflow.log_text(str(cm), "confusion_matrix.txt")
    
    return metrics


def train_logistic_regression(X_train, X_test, y_train, y_test, params=None):
    """Train Logistic Regression with MLflow tracking."""
    
    if params is None:
        params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'lbfgs'
        }
    
    with mlflow.start_run(run_name="Logistic_Regression_Baseline"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "LogisticRegression")
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Log metrics
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print(f"\n{'='*60}")
        print("LOGISTIC REGRESSION RESULTS")
        print(f"{'='*60}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        return model, metrics


def train_random_forest(X_train, X_test, y_train, y_test, params=None):
    """Train Random Forest with MLflow tracking."""
    
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 5,
            'min_samples_split': 10,
            'random_state': 42
        }
    
    with mlflow.start_run(run_name="Random_Forest_Baseline"):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Log metrics
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        # Log feature importances
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        mlflow.log_text(feature_importance.to_string(), "feature_importance.txt")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Print results
        print(f"\n{'='*60}")
        print("RANDOM FOREST RESULTS")
        print(f"{'='*60}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        return model, metrics


def train_random_forest_tuned(X_train, X_test, y_train, y_test):
    """Train Random Forest with tuned hyperparameters."""
    
    params = {
        'n_estimators': 200,
        'max_depth': 7,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42
    }
    
    with mlflow.start_run(run_name="Random_Forest_Tuned"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("tuning_method", "manual")
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\n{'='*60}")
        print("RANDOM FOREST TUNED RESULTS")
        print(f"{'='*60}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        return model, metrics


def train_decision_tree(X_train, X_test, y_train, y_test):
    """Train Decision Tree with MLflow tracking."""
    
    params = {
        'max_depth': 5,
        'min_samples_split': 20,
        'random_state': 42
    }
    
    with mlflow.start_run(run_name="Decision_Tree"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "DecisionTreeClassifier")
        
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\n{'='*60}")
        print("DECISION TREE RESULTS")
        print(f"{'='*60}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        return model, metrics


def train_xgboost(X_train, X_test, y_train, y_test):
    """Train XGBoost with MLflow tracking."""
    
    params = {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    with mlflow.start_run(run_name="XGBoost"):
        mlflow.log_params(params)
        mlflow.log_param("model_type", "XGBClassifier")
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        mlflow.sklearn.log_model(model, "model")
        
        print(f"\n{'='*60}")
        print("XGBOOST RESULTS")
        print(f"{'='*60}")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name}: {metric_value:.4f}")
        
        return model, metrics


def main():
    """Main training pipeline with MLflow tracking."""
    print("="*60)
    print("MLFLOW EXPERIMENT TRACKING - TITANIC SURVIVAL PREDICTION")
    print("="*60)
    
    # Setup MLflow
    setup_mlflow()
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = load_raw_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train multiple models
    all_results = []
    
    # 1. Logistic Regression
    print("\n" + "="*60)
    print("Training Logistic Regression...")
    print("="*60)
    lr_model, lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)
    all_results.append({**{'model': 'Logistic Regression'}, **lr_metrics})
    
    # 2. Decision Tree
    print("\n" + "="*60)
    print("Training Decision Tree...")
    print("="*60)
    dt_model, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test)
    all_results.append({**{'model': 'Decision Tree'}, **dt_metrics})
    
    # 3. Random Forest (Baseline)
    print("\n" + "="*60)
    print("Training Random Forest (Baseline)...")
    print("="*60)
    rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
    all_results.append({**{'model': 'Random Forest'}, **rf_metrics})
    
    # 4. Random Forest (Tuned)
    print("\n" + "="*60)
    print("Training Random Forest (Tuned)...")
    print("="*60)
    rf_tuned_model, rf_tuned_metrics = train_random_forest_tuned(X_train, X_test, y_train, y_test)
    all_results.append({**{'model': 'Random Forest Tuned'}, **rf_tuned_metrics})
    
    # 5. XGBoost
    print("\n" + "="*60)
    print("Training XGBoost...")
    print("="*60)
    xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test)
    all_results.append({**{'model': 'XGBoost'}, **xgb_metrics})
    
    # Create comparison table
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('accuracy', ascending=False)
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPARISON")
    print(f"{'='*60}")
    print(results_df.to_string(index=False))
    
    # Save results
    Path("reports").mkdir(exist_ok=True)
    results_df.to_csv("reports/mlflow_experiments.csv", index=False)
    print("\n✅ Results saved to reports/mlflow_experiments.csv")
    
    # Print MLflow UI instructions
    print(f"\n{'='*60}")
    print("✅ ALL EXPERIMENTS LOGGED TO MLFLOW")
    print(f"{'='*60}")
    print("\nTo view experiments in MLflow UI:")
    print("1. Run: mlflow ui")
    print("2. Open browser: http://localhost:5000")
    print("3. Compare experiments and take screenshots for report")
    
    print(f"\n{'='*60}")
    print("✅ DAY 2 COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()