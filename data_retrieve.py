import mlflow
import hashlib
from mlflow.tracking import MlflowClient
import pandas as pd

#Instantiate the Client
client = MlflowClient()

# Get the latest version of model
model_name = "Logistic Regression"
latest_versions = client.get_latest_versions(name = model_name, stages = None)
latest_version = latest_versions[0].version
print(f"Latest version for model: {model_name} | Version: {latest_version}")


# Setting the datahash with the model version
# client.set_model_version_tag(
#     name = model_name,
#     version = latest_version,
#     key = "data_hash",
#     value = data_hash
# )


# Get the data hash for that model
mv = client.get_model_version(
    name = model_name,
    version = latest_version
)

dataset_hash = mv.tags["data_hash"]
print(f"Data hash for model: {model_name} | Version: {latest_version} | Data Hash: {dataset_hash}")

# Get the run id for that data hash
experiment_id = "2"
runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f"tags.data_hash = '{dataset_hash}'"
)

run_id = runs[0].info.run_id
print(f"Run id: {run_id}")

# Create a local path for the data artifact of the run
local_path = client.download_artifacts(
    run_id = run_id,
    path = "train_dataset.csv"
)

df = pd.read_csv(local_path)
print(df)