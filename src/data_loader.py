import io
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from config import RAW_DATA_PATH, ALL_FEATURES, TARGET_COLUMN

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

PRIMARY_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
FALLBACK_MIRROR_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Heart%20Disease%20Cleveland.csv"


def download_uci_dataset(dest_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Download the official UCI Cleveland dataset or fetch from fallback mirror."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {"User-Agent": "Mozilla/5.0"}
    df = None

    try:
        req = urllib.request.Request(PRIMARY_UCI_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(content), header=None, names=COLUMN_NAMES, na_values="?")
            print("Successfully loaded dataset from UCI repository.")
    except Exception as e:
        print(f"Warning: Primary UCI URL download failed ({e}). Trying fallback mirror...")
        try:
            req = urllib.request.Request(FALLBACK_MIRROR_URL, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                content = response.read().decode("utf-8")
                # Fallback mirror might have headers; check first line
                first_line = content.splitlines()[0]
                if "age" in first_line.lower():
                    df = pd.read_csv(io.StringIO(content), na_values="?")
                    # Rename columns to standard lowercase names if needed
                    df.columns = [c.strip().lower() for c in df.columns]
                    # Map 'condition' or 'num' or 'target'
                    if "condition" in df.columns:
                        df.rename(columns={"condition": "target"}, inplace=True)
                    elif "num" in df.columns:
                        df.rename(columns={"num": "target"}, inplace=True)
                else:
                    df = pd.read_csv(io.StringIO(content), header=None, names=COLUMN_NAMES, na_values="?")
            print("Successfully loaded dataset from fallback mirror.")
        except Exception as fallback_e:
            raise RuntimeError(f"Failed to fetch dataset from all sources: {fallback_e}")

    # Standardize target to binary (0: healthy, 1: disease present)
    df[TARGET_COLUMN] = (df[TARGET_COLUMN] > 0).astype(int)

    # Cast numeric columns appropriately
    for col in ["age", "trestbps", "chol", "thalach", "oldpeak"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Save to disk
    df.to_csv(dest_path, index=False)
    print(f"Raw dataset saved to: {dest_path} (Shape: {df.shape})")
    return df


def load_raw_data(filepath: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw dataset from disk, downloading if absent."""
    if not filepath.exists():
        print(f"Dataset not found at {filepath}. Downloading now...")
        return download_uci_dataset(filepath)

    df = pd.read_csv(filepath)
    # Ensure standard schema
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that dataframe adheres to expected schema and ranges."""
    missing_cols = [col for col in COLUMN_NAMES if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}")

    if not set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
        raise ValueError(f"Target column must be binary (0 or 1), got {df[TARGET_COLUMN].unique()}")

    if df.shape[0] < 200:
        raise ValueError(f"Dataset unexpectedly small: {df.shape[0]} rows")

    return True


if __name__ == "__main__":
    df = load_raw_data()
    print("Schema Validation: OK")
    print(f"Dataset summary:\n- Rows: {len(df)}\n- Columns: {len(df.columns)}")
    print(f"- Target distribution:\n{df['target'].value_counts(normalize=True).to_dict()}")
    print(f"- Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
