import pandas as pd
import numpy as np
import datetime
import joblib
from sklearn.linear_model import LinearRegression
from evidently.report import Report
from evidently.metric_preset import TargetDriftPreset, DataDriftPreset
from evidently.metrics import DataDriftTable

# ------------------------------------------------------
# 1. Load datasets (reference and current)
# ------------------------------------------------------
print("Loading Auto MPG datasets...")

reference_data = pd.read_csv("./auto-mpg.csv")
current_data = pd.read_csv("./auto_mpg_drifted.csv")

# Replace '?' with NaN and convert to numeric where needed
reference_data.replace("?", np.nan, inplace=True)
current_data.replace("?", np.nan, inplace=True)

features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration']
target = 'mpg'

# Ensure numeric types for features
for col in features + [target]:
    reference_data[col] = pd.to_numeric(reference_data[col], errors='coerce')
    current_data[col] = pd.to_numeric(current_data[col], errors='coerce')

# ------------------------------------------------------
# 2. Prepare and train model
# ------------------------------------------------------
print("\nTraining a simple linear regression model...")

X_ref = reference_data[features].dropna()
y_ref = reference_data.loc[X_ref.index, target]

model = LinearRegression()
model.fit(X_ref, y_ref)

# Save the model
model_filename = 'auto_mpg_model.joblib'
joblib.dump(model, model_filename)
print(f"Model saved to {model_filename}")

# ------------------------------------------------------
# 3. Predictions
# ------------------------------------------------------
reference_data = reference_data.dropna(subset=features + [target])
reference_data['prediction'] = model.predict(reference_data[features])

current_data = current_data.dropna(subset=features + [target])
current_data['prediction'] = model.predict(current_data[features])


# ------------------------------------------------------
# 4. Model drift report
# ------------------------------------------------------
print("\nGenerating Evidently Model Performance report...")
model_report = Report(metrics=[TargetDriftPreset()])
model_report.run(reference_data=reference_data, current_data=current_data)
model_report_filename = f'model_performance_report_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.html'

model_report.save_html(model_report_filename)
print(f"Model drift report saved to {model_report_filename}")


# ------------------------------------------------------
# 5. Data drift dashboard (Summary + Detailed Table)
# ------------------------------------------------------
print("\nGenerating Evidently Data Drift dashboard...")
data_drift_report = Report(metrics=[
    DataDriftPreset(),
    DataDriftTable()
])
data_drift_report.run(reference_data=reference_data, current_data=current_data)
data_drift_report_filename = f"data_drift_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
data_drift_report.save_html(data_drift_report_filename)
print(f"Data drift dashboard saved to {data_drift_report_filename}")


# ------------------------------------------------------
# 6. Drift detection function
# ------------------------------------------------------
def check_for_drift(report_json, threshold=0.1):
    for metric in report_json.get('metrics', []):
        if metric.get('metric') == 'DatasetDriftMetric':
            drift_share = metric.get('result', {}).get('drift_share')
            if drift_share and drift_share > threshold:
                return True
    return False


# ------------------------------------------------------
# 7. Local alert
# ------------------------------------------------------
def local_alert(drift_detected):
    if drift_detected:
        print("\n!!! ALERT: Data Drift Detected in Auto MPG Dataset !!!")
        print("Please check the generated HTML reports for details.")
    else:
        print("\nNo significant data drift detected in Auto MPG Dataset.")

# Run drift check
report_json_data = model_report.as_dict()
drift_detected = check_for_drift(report_json_data)
local_alert(drift_detected)
