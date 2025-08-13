import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Load the saved model
with open("autompg_model.bin", "rb") as f:
    model = pickle.load(f)

# Function to prepare features for prediction
def prepare_features(data):
    return pd.DataFrame([{
        "cylinders": data["cylinders"],
        "displacement": data["displacement"],
        "horsepower": data["horsepower"],
        "weight": data["weight"],
        "acceleration": data["acceleration"],
        "model_year": data["model_year"],
        "origin": data["origin"]
    }])

# Prediction function
def predict(features_df):
    preds = model.predict(features_df)
    return float(preds[0])

# Flask app
app = Flask("auto-mpg-prediction")

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    input_data = request.get_json()

    features_df = prepare_features(input_data)
    prediction = predict(features_df)

    result = {
        "predicted_mpg": prediction
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
