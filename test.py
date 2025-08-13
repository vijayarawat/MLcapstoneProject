import requests

sample_car = {
    "cylinders": 4,
    "displacement": 140.0,
    "horsepower": 90.0,
    "weight": 2264.0,
    "acceleration": 15.5,
    "model_year": 82,
    "origin": 1
}

url = "http://localhost:9696/predict"
response = requests.post(url, json=sample_car)

print("Prediction response:", response.json())
