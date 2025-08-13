import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import pickle

#  Load Dataset

df = pd.read_csv("./auto-mpg.csv")

print("First 5 records:")
print(df.head())

print("\nMissing values per column:")
print(df.isna().sum())

# Number of records and columns
rows, cols = df.shape
print(f"\nNumber of records: {rows}")
print(f"Number of columns: {cols}")

print("\nData types of each field:")
print(df.info())


# Data Cleaning

df.replace("?", np.nan, inplace=True)

df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

print("\nData info after cleaning:")
print(df.info())


data = df.drop("car name", axis=1)


#Feature & Target Split

X = data.drop('mpg', axis=1)
y = data['mpg']


#Preprocessing

numeric_features = ['displacement', 'horsepower', 'weight', 'acceleration']
categorical_features = ['cylinders', 'model_year', 'origin']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Model Dictionary

models = {
    'LinearRegression': LinearRegression(),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'KNeighbors': KNeighborsRegressor()
}


#  MLflow Tracking Setup

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("AutoMPG_Regression")

best_rmse = float('inf')
best_model_name = None
best_model = None
best_run_id = None


# Model Training & Logging

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        run_id = run.info.run_id

        # Create pipeline (Preprocessing + Model)
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)

        print(f"{name} - RMSE: {rmse:.3f}, MAE: {mae:.3f}")

        # Log parameters & metrics
        mlflow.log_param("model_name", name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)

        # Log model with signature
        signature = infer_signature(X_train, pipeline.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature
        )

        # Track best model
        if rmse < best_rmse:
            best_rmse = rmse
            best_model_name = name
            best_model = pipeline
            best_run_id = run_id
            

# Register Best Model in MLflow

client = MlflowClient()
model_uri = f"runs:/{best_run_id}/model"
model_name = "AutoMPG_BestModel"
model_version = mlflow.register_model(model_uri, model_name)

print(f"\nBest model ({best_model_name}) registered as {model_name}, version {model_version.version}")


#Save Best Model Locally

def save_model_locally(model, filename):
    with open(filename, 'wb') as f_out:
        pickle.dump(model, f_out)

save_model_locally(best_model, "best_model.bin")
print(f"Best model saved locally as best_model.bin")
