import joblib

model = joblib.load("api/model/model.pkl")

# Check if model has feature_names_in_ attribute
if hasattr(model, "feature_names_in_"):
    print("Model expects these features:")
    print(list(model.feature_names_in_))
else:
    print("Model doesn't store feature names")
    print("Try making a prediction to see what it expects")
