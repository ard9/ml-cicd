from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.model import Predictor

app = FastAPI(title="Iris Classifier API", version="1.0.0")

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


class PredictRequest(BaseModel):
    features: List[float]  # [sepal_length, sepal_width, petal_length, petal_width]


class PredictResponse(BaseModel):
    predicted_class: str
    predicted_class_index: int
    probabilities: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    predictor = get_predictor()
    return predictor.metadata


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    predictor = get_predictor()
    try:
        result = predictor.predict(request.features)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return result
