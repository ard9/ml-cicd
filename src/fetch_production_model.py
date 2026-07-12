import os
from pathlib import Path
import mlflow
import mlflow.sklearn
import joblib
from mlflow.tracking import MlflowClient

MODEL_NAME = "iris-classifier"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    mv = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_DIR / "model.joblib")
    client.download_artifacts(mv.run_id, "metadata.json", str(MODEL_DIR))

    print(f"Fetched Production model v{mv.version} (run {mv.run_id}) -> {MODEL_DIR}")


if __name__ == "__main__":
    main()