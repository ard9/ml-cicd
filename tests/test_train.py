from src.train import train_and_evaluate, MIN_ACCURACY


def test_training_produces_metadata_with_expected_keys():
    _model, metadata = train_and_evaluate()
    assert "metrics" in metadata
    assert "accuracy" in metadata["metrics"]
    assert "class_names" in metadata
    assert len(metadata["class_names"]) == 3


def test_training_meets_quality_gate():
    """This is the test that mirrors CI's quality gate.
    If this fails, the pipeline should stop before shipping."""
    _model, metadata = train_and_evaluate()
    assert metadata["metrics"]["accuracy"] >= MIN_ACCURACY


def test_model_can_predict_on_test_split():
    model, metadata = train_and_evaluate()
    sample = [5.1, 3.5, 1.4, 0.2]  # a typical setosa
    prediction = model.predict([sample])
    assert prediction[0] in [0, 1, 2]
