"""
Compares the newest registered model version against the current
Production version, and promotes the winner.
"""
import os
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "iris-classifier"
PRIMARY_METRIC = "accuracy"


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    candidate = client.get_latest_versions(MODEL_NAME, stages=["None"])[0]
    candidate_run = client.get_run(candidate.run_id)
    candidate_score = candidate_run.data.metrics[PRIMARY_METRIC]
    print(f"Candidate: v{candidate.version}, {PRIMARY_METRIC}={candidate_score}")

    prod = client.get_latest_versions(MODEL_NAME, stages=["Production"])

    if not prod:
        print("No Production model yet. Promoting candidate directly.")
        client.transition_model_version_stage(
            MODEL_NAME, candidate.version, "Production"
        )
        return

    prod_run = client.get_run(prod[0].run_id)
    prod_score = prod_run.data.metrics[PRIMARY_METRIC]
    print(f"Current Production: v{prod[0].version}, {PRIMARY_METRIC}={prod_score}")

    if candidate_score > prod_score:
        print("Candidate wins. Promoting, archiving previous Production.")
        client.transition_model_version_stage(
            MODEL_NAME, candidate.version, "Production",
            archive_existing_versions=True,
        )
    else:
        print("Candidate is not better. Marking as Staging for reference.")
        client.transition_model_version_stage(
            MODEL_NAME, candidate.version, "Staging"
        )


if __name__ == "__main__":
    main()
