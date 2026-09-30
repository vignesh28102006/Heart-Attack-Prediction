import pytest
import numpy as np
import pandas as pd
from config import PREPROCESSOR_PATH
from src.data_loader import load_raw_data
from src.preprocessor import load_preprocessor, prepare_and_split_data


def test_preprocessing_output_shape():
    """Verify preprocessor produces correct matrix shapes without NaNs."""
    df = load_raw_data()
    splits = prepare_and_split_data(df, test_size=0.2, random_state=42, save_artifacts=False)

    X_train_trans = splits["X_train_trans"]
    X_test_trans = splits["X_test_trans"]

    assert X_train_trans.shape[0] == len(splits["y_train"])
    assert X_test_trans.shape[0] == len(splits["y_test"])
    assert X_train_trans.shape[1] == X_test_trans.shape[1]

    # No NaN in transformed data
    assert not np.isnan(X_train_trans).any(), "NaN values detected in X_train_trans"
    assert not np.isnan(X_test_trans).any(), "NaN values detected in X_test_trans"


def test_preprocessor_serialization():
    """Verify serialized preprocessor loads and transforms single instance accurately."""
    assert PREPROCESSOR_PATH.exists(), "Serialized preprocessor does not exist"
    preprocessor = load_preprocessor()

    sample = pd.DataFrame([{
        "age": 60,
        "sex": 1,
        "cp": 0,
        "trestbps": 140,
        "chol": 260,
        "fbs": 0,
        "restecg": 0,
        "thalach": 145,
        "exang": 1,
        "oldpeak": 2.0,
        "slope": 1,
        "ca": 1,
        "thal": 2,
    }])

    trans = preprocessor.transform(sample)
    assert trans.shape[0] == 1
    assert trans.shape[1] > 14  # Expanded due to one-hot encoding
    assert not np.isnan(trans).any()
