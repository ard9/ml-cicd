import joblib
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"


def test_model_file_exists():
    assert MODEL_PATH.exists(), "مدل fetch نشده — fetch_production_model.py اجرا شده؟"


def test_predictions_are_valid_classes():
    model = joblib.load(MODEL_PATH)
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    pred = model.predict(sample)
    assert pred[0] in [0, 1, 2]


def test_sanity_accuracy_on_full_dataset():
    model = joblib.load(MODEL_PATH)
    data = load_iris()
    acc = (model.predict(data.data) == data.target).mean()
    assert acc > 0.85, f"accuracy پایین‌تر از انتظار: {acc}"