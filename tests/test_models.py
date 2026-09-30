import pytest
import numpy as np
import pandas as pd
from config import (
    LOGISTIC_REGRESSION_PATH,
    METRICS_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
)
from src.evaluate import load_all_models
from src.preprocessor import load_preprocessor


def test_models_exist():
    """Verify that all 3 serialized models exist on disk."""
    assert LOGISTIC_REGRESSION_PATH.exists(), "Logistic Regression model artifact missing"
    assert RANDOM_FOREST_PATH.exists(), "Random Forest model artifact missing"
    assert XGBOOST_PATH.exists(), "XGBoost model artifact missing"


def test_models_inference():
    """Verify all 3 models generate predictions and valid probability distributions."""
    models = load_all_models()
    preprocessor = load_preprocessor()

    sample_dict = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1,
    }
    input_df = pd.DataFrame([sample_dict])
    X_sample = preprocessor.transform(input_df)

    for name, model in models.items():
        pred = model.predict(X_sample)
        assert pred[0] in [0, 1], f"{name} predicted invalid class: {pred[0]}"

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_sample)
            assert proba.shape == (1, 2), f"{name} proba shape mismatch"
            assert 0.0 <= proba[0, 1] <= 1.0, f"{name} proba out of bounds"
            assert np.isclose(np.sum(proba), 1.0), f"{name} probabilities do not sum to 1.0"


def test_metrics_file_validity():
    """Verify evaluation metrics JSON contains expected structure and non-zero recall."""
    import json
    assert METRICS_PATH.exists(), "evaluation_metrics.json missing"
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "models" in data
    assert "Logistic Regression" in data["models"]
    assert "Random Forest" in data["models"]
    assert "XGBoost" in data["models"]

    for m_name, m_stats in data["models"].items():
        assert m_stats["recall"] >= 0.70, f"{m_name} recall unexpectedly low: {m_stats['recall']}"
        assert m_stats["accuracy"] >= 0.70, f"{m_name} accuracy unexpectedly low: {m_stats['accuracy']}"
