import pandas as pd
import sys
import json
sys.path.insert(0, 'src')
from evidently import Report
from evidently.presets import DataDriftPreset
from data.load_data import load_raw_data
from data.preprocessing import preprocess_data

# Load the real dataset
df = load_raw_data()
X, y = preprocess_data(df)

# Use first 600 rows as reference (training data)
reference = X.iloc[:600]

# Use last 291 rows as current (production data)
current = X.iloc[600:]

# Create drift report
report = Report([DataDriftPreset()])
result = report.run(reference_data=reference, current_data=current)

# Save as JSON
result_dict = result.dict()
with open("monitoring/drift_report.json", "w") as f:
    json.dump(result_dict, f, indent=2, default=str)

print("✅ Drift report saved to monitoring/drift_report.json")
print("\nDrift summary:")
for metric in result_dict.get("metrics", []):
    print(metric)