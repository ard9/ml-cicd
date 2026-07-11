"""
Training pipeline for the Iris classifier.

This script represents the "MLOps" heart of the project:
1. Load data
2. Train model
3. Evaluate model
4. Save model + metadata (versioned)
5. Enforce a quality gate: if accuracy is below MIN_ACCURACY, exit with
   a non-zero code so the CI/CD pipeline stops and does NOT ship a bad model.
"""

import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

MIN_ACCURACY = 0.90  # quality gate threshold


def train_and_evaluate(random_state: int = 42):
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="macro")

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1), 4),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_names": list(data.feature_names),
        "class_names": list(data.target_names),
        "metrics": metrics,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "min_accuracy_gate": MIN_ACCURACY,
    }

    return model, metadata


def save_artifacts(model, metadata: dict):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))


def main():
    print("Starting training run...")
    model, metadata = train_and_evaluate()

    print(json.dumps(metadata, indent=2))

    accuracy = metadata["metrics"]["accuracy"]
    if accuracy < MIN_ACCURACY:
        print(
            f"QUALITY GATE FAILED: accuracy {accuracy} < required {MIN_ACCURACY}. "
            "Refusing to save/ship this model."
        )
        sys.exit(1)

    save_artifacts(model, metadata)
    print(f"QUALITY GATE PASSED. Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
