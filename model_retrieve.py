import mlflow
import hashlib
from mlflow.tracking import MlflowClient

#Instantiate the Client
client = MlflowClient()

# Get the latest version of model
model_name = "Logistic Regression"
latest_versions = client.get_latest_versions(name = model_name, stages = None)
latest_version = latest_versions[0].version
print(f"Latest version for model: {model_name} | Version: {latest_version}")

loaded_model_uri = f"models:/{model_name}/{latest_version}"
loaded_model = mlflow.sklearn.load_model(loaded_model_uri)
print(f"Loaded model: {loaded_model}")