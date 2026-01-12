import mlflow
from mlflow.tracking import MlflowClient
from typing import List, Tuple

client = MlflowClient()


def find_best_model(candidate_models: List[str]) -> Tuple[str, float, int]:
    """Finding best model"""
    best_model = None
    best_version = None
    best_accuracy = -1
    
    for model_name in candidate_models:
        versions = client.get_latest_versions(model_name)
    
        for v in versions:
            run = client.get_run(v.run_id)
            accuracy = run.data.metrics.get("accuracy")
    
            if accuracy is not None and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
                best_version = v.version

    return  best_model, best_accuracy, best_version


candidate_models = [
    "Logistic Regression",
    "Random Forest"
]
best_model, best_accuracy, best_version = find_best_model(candidate_models)
print(f"Best model: {best_model}")
print(f"Best accuracy: {best_accuracy}")
print(f"Best version: {best_version}")



def promote_model(new_model: str, new_version: int, old_model: str, old_version: int):
    """Promoting new model to prod"""
    client.set_registered_model_alias(
        name=new_model,
        alias="prod",
        version=new_version
    )
    client.set_registered_model_alias(
        name=old_model,
        alias="challenger",
        version=old_version
    )
    client.delete_registered_model_alias(
        name=old_model,
        alias="prod"
    )

new_model = "Logistic Regression"
new_version = 3
old_model = "Random Forest"
old_version = 1
promote_model(new_model, new_version, old_model, old_version)



