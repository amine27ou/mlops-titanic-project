import mlflow
import joblib

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('titanic-survival')

with mlflow.start_run(run_name='LogisticRegression_registered'):
    model = joblib.load('./api/model/model.pkl')
    mlflow.log_metric('accuracy', 0.8045)
    mlflow.log_param('model_type', 'LogisticRegression')
    mlflow.sklearn.log_model(
        model,
        artifact_path='model',
        registered_model_name='titanic-survival-model'
    )
    print('Done! Model registered.')