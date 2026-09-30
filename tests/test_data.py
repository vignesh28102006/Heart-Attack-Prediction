import pytest
import pandas as pd
from pathlib import Path
from config import RAW_DATA_PATH, ALL_FEATURES, TARGET_COLUMN
from src.data_loader import load_raw_data, validate_schema


def test_raw_data_exists_and_loads():
    """Verify raw dataset exists and can be loaded as DataFrame."""
    assert RAW_DATA_PATH.exists(), f"Raw data not found at {RAW_DATA_PATH}"
    df = load_raw_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 200


def test_schema_columns():
    """Verify all 13 features plus target column are present."""
    df = load_raw_data()
    for col in ALL_FEATURES + [TARGET_COLUMN]:
        assert col in df.columns, f"Expected column {col} missing from raw dataset"


def test_target_binary_values():
    """Verify target column contains only 0 (healthy) and 1 (disease)."""
    df = load_raw_data()
    unique_targets = set(df[TARGET_COLUMN].dropna().unique())
    assert unique_targets.issubset({0, 1}), f"Unexpected target classes: {unique_targets}"


def test_validate_schema_raises_on_invalid():
    """Verify validate_schema catches invalid schemas."""
    bad_df = pd.DataFrame({"age": [50], "sex": [1]})
    with pytest.raises(ValueError):
        validate_schema(bad_df)
