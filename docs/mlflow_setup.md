# MLflow Tracking Setup

## Overview

MLflow is used for experiment tracking, logging parameters, metrics, and model artifacts.

## Configuration

**Tracking URI:** `file:./mlruns` (local filesystem)  
**Experiment Name:** `titanic-survival-prediction`

## Experiments Logged

| Experiment | Model Type | Key Parameters |
|------------|-----------|----------------|
| Logistic_Regression_Baseline | LogisticRegression | max_iter=1000, solver=lbfgs |
| Decision_Tree | DecisionTreeClassifier | max_depth=5, min_samples_split=20 |
| Random_Forest_Baseline | RandomForestClassifier | n_estimators=100, max_depth=5 |
| Random_Forest_Tuned | RandomForestClassifier | n_estimators=200, max_depth=7 |
| XGBoost | XGBClassifier | n_estimators=100, max_depth=3, lr=0.1 |

## Metrics Tracked

For each experiment, the following metrics are logged:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC Score
- Confusion Matrix

## Artifacts Stored

- Trained model (pickle format)
- Feature importance (for tree-based models)
- Confusion matrix (text file)

## Viewing Results

```bash
mlflow ui
```

Then navigate to http://localhost:5000

## Best Model

**Model:** Random Forest Tuned  
**Accuracy:** XX.XX%  
**F1 Score:** XX.XX%

Selected for deployment based on highest accuracy and F1 score.