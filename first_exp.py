import numpy as np
import pandas as pd
import mlflow
import hashlib
from mlflow.tracking import MlflowClient
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import warnings


X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8, 
                           weights=[0.9, 0.1], flip_y=0, random_state=42)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Save locally for artifact logging
train_df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
train_df["target"] = y_train
#train_path = "train_dataset.csv"
#train_df.to_csv(train_path, index=False)

def hash_array(arr):
    """Compute a hash for a numpy array"""
    return hashlib.md5(arr.tobytes()).hexdigest()


def hash_dataframe(df: pd.DataFrame) -> str:
    return hashlib.md5(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

data_hash = hash_dataframe(train_df)

params = {
"solver": "lbfgs",
"max_iter": 100000,
#"multi_class": "auto",
"random_state":42
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

report = classification_report(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

#ec2 instance
mlflow.set_tracking_uri(uri="http://35.154.73.149:5000/")
mlflow.set_experiment("First experiment new")

train_path = "train_dataset.csv"

# Save locally for artifact logging
train_df = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
train_df["target"] = y_train
train_path = "train_dataset.csv"
train_df.to_csv(train_path, index=False)

with mlflow.start_run(run_name = "logistic_regression_v3"):
    mlflow.log_params(params)
    mlflow.log_metrics(
        {
            "accuracy": report_dict["accuracy"],
            "recall_class_0": report_dict["0"]["recall"],
            "recall_class_1": report_dict["1"]["recall"],
            "precision_class_0": report_dict["0"]["precision"],
            "precision_class_1": report_dict["1"]["precision"],
            "f1_score_class_0": report_dict["0"]["f1-score"],
            "f1_score_class_1": report_dict["1"]["f1-score"],
            
        }
    )

    #logging data as artifact with unique data hash tag
    mlflow.set_tag("data_hash", data_hash)
    mlflow.log_artifact(train_path)

    #registering model
    mlflow.sklearn.log_model(lr, 
                             name = "model",
                             registered_model_name = "Logistic Regression")

#connecting the datahash to the model
client = MlflowClient()
client.set_model_version_tag(
    name = "Logistic Regression",
    version = "1",
    key = "data_hash",
    value = data_hash
)
