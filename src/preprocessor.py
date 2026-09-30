import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PREPROCESSOR_PATH,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def create_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for numerical scaling and categorical encoding."""
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, NUMERICAL_FEATURES),
            ("cat", cat_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Retrieve output feature names after column transformation."""
    feature_names = []
    
    # Numerical names
    feature_names.extend(NUMERICAL_FEATURES)
    
    # Categorical one-hot encoded names
    cat_encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    feature_names.extend(cat_feature_names.tolist())
    
    return feature_names


def prepare_and_split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    save_artifacts: bool = True,
):
    """Split data into train/test sets, fit preprocessor on train only, transform both."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()

    # Stratified split to preserve class imbalance ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = create_preprocessor()

    # CRITICAL: Fit ONLY on training set to eliminate data leakage
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = get_feature_names(preprocessor)

    # Convert to DataFrames for inspection & CSV persistence
    X_train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names)

    if save_artifacts:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        print(f"Preprocessor serialized to: {PREPROCESSOR_PATH}")

        # Save processed train and test splits
        train_df = X_train_df.copy()
        train_df[TARGET_COLUMN] = y_train.values
        train_df.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)

        test_df = X_test_df.copy()
        test_df[TARGET_COLUMN] = y_test.values
        test_df.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
        print(f"Processed train/test datasets saved to: {PROCESSED_DATA_DIR}")

    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_train_trans": X_train_transformed,
        "X_test_trans": X_test_transformed,
        "feature_names": feature_names,
        "preprocessor": preprocessor,
    }


def load_preprocessor(path: Path = PREPROCESSOR_PATH) -> ColumnTransformer:
    """Load serialized preprocessor from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {path}. Run data pipeline first.")
    return joblib.load(path)


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    raw_df = load_raw_data()
    splits = prepare_and_split_data(raw_df)
    print("Preprocessed shapes:")
    print(f"- X_train: {splits['X_train_trans'].shape}")
    print(f"- X_test:  {splits['X_test_trans'].shape}")
    print(f"- Transformed feature count: {len(splits['feature_names'])}")
