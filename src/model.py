"""
Thin wrapper around the trained model so the API (and tests) don't need
to know anything about how the model was trained or where it's stored.
"""

import json
from pathlib import Path
from typing import List

import joblib

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"


class Predictor:
    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run `python src/train.py` first."
            )
        self.model = joblib.load(MODEL_PATH)
        self.metadata = json.loads(METADATA_PATH.read_text())
        self.class_names = self.metadata["class_names"]
        self.feature_names = self.metadata["feature_names"]

    def predict(self, features: List[float]) -> dict:
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"Expected {len(self.feature_names)} features "
                f"({self.feature_names}), got {len(features)}"
            )
        pred_idx = int(self.model.predict([features])[0])
        proba = self.model.predict_proba([features])[0].tolist()

        return {
            "predicted_class": self.class_names[pred_idx],
            "predicted_class_index": pred_idx,
            "probabilities": {
                self.class_names[i]: round(p, 4) for i, p in enumerate(proba)
            },
        }
