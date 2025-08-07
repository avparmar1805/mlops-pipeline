import argparse
import mlflow
import os
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def load_data(path, target_col="MedHouseVal"):
    """
    Load the California Housing CSV and split into features X and target y.
    """
    df = pd.read_csv(path)
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def get_model(model_name):
    """
    Return an untrained scikit-learn model instance.
    """
    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
    }
    return models[model_name]

def evaluate(y_true, y_pred):
    """
    Compute Mean Squared Error for regression.
    """
    return mean_squared_error(y_true, y_pred)

def main(args):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("California-Housing")

    # 1) Load & split the data
    X_train, X_test, y_train, y_test = load_data(args.data_path, args.target_col)

    # 2) Initialize an MLflow run
    with mlflow.start_run(run_name=args.model_name) as run:
        # 3) Instantiate and train the model
        model = get_model(args.model_name)
        model.fit(X_train, y_train)

        # 4) Predict on test set and evaluate
        preds = model.predict(X_test)
        mse = evaluate(y_test, preds)

        # 5) Log parameters, metrics, and the model artifact
        mlflow.log_param("model_name", args.model_name)
        mlflow.log_param("random_state", 42)
        # log any hyperparameters specific to the model
        if args.model_name == "random_forest":
            mlflow.log_param("n_estimators", 100)

        mlflow.log_metric("mse", mse)

        # Save the trained model in MLflow’s artifact store
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="CaliforniaHousingRegressor"  # auto-register best later
        )

        client = MlflowClient()
        client.transition_model_version_stage(
            name="CaliforniaHousingRegressor",
            version=1,  # Or dynamically get the latest version
            stage="Production"
        )
        print(f"Run ID: {run.info.run_id} | Model: {args.model_name} | MSE: {mse:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train California Housing regression models with MLflow")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/california_housing.csv",
        help="Path to the CSV file (default: data/raw/california_housing.csv)"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="MedHouseVal",
        help="Name of the target column (default: MedHouseVal)"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["linear_regression", "random_forest"],
        required=True,
        help="Which regression model to train"
    )
    args = parser.parse_args()
    main(args)
