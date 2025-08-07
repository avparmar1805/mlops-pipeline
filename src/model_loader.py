import mlflow
from mlflow.tracking import MlflowClient

def load_model(model_name: str, stage: str = "Production"):
    """
    Fetches the latest model version in the given stage from MLflow Model Registry
    and returns a loaded sklearn model.
    """
    client = MlflowClient()
    # Get latest version in “Production” (or whichever stage you registered)
    mv = client.get_latest_versions(model_name, stages=[stage])[0]
    model_uri = f"models:/{model_name}/{mv.version}"
    return mlflow.sklearn.load_model(model_uri)
