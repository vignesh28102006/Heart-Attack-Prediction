import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
import xgboost as xgb

from config import (
    CV_FOLDS,
    LOGISTIC_REGRESSION_PATH,
    MODELS_DIR,
    RANDOM_FOREST_PATH,
    RANDOM_STATE,
    XGBOOST_PATH,
)


def get_models(scale_pos_weight: float = 1.0) -> dict:
    """Instantiate the 3 required classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            solver="liblinear",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        ),
    }
    return models


def evaluate_cross_validation(models: dict, X_train: np.ndarray, y_train: pd.Series) -> dict:
    """Perform 5-fold Stratified Cross Validation focusing on Recall."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    scoring = ["recall", "accuracy", "precision", "f1", "roc_auc"]

    print("\n--- 5-Fold Stratified Cross-Validation Results (Train Split) ---")
    for name, model in models.items():
        scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )

        cv_results[name] = {
            "cv_recall_mean": float(np.mean(scores["test_recall"])),
            "cv_recall_std": float(np.std(scores["test_recall"])),
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_accuracy_std": float(np.std(scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(scores["test_precision"])),
            "cv_precision_std": float(np.std(scores["test_precision"])),
            "cv_f1_mean": float(np.mean(scores["test_f1"])),
            "cv_f1_std": float(np.std(scores["test_f1"])),
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "cv_roc_auc_std": float(np.std(scores["test_roc_auc"])),
        }

        print(
            f"[{name}]\n"
            f"  Recall:    {cv_results[name]['cv_recall_mean']:.4f} (±{cv_results[name]['cv_recall_std']:.4f})  <-- PRIMARY\n"
            f"  ROC-AUC:   {cv_results[name]['cv_roc_auc_mean']:.4f} (±{cv_results[name]['cv_roc_auc_std']:.4f})\n"
            f"  F1-Score:  {cv_results[name]['cv_f1_mean']:.4f} (±{cv_results[name]['cv_f1_std']:.4f})\n"
            f"  Accuracy:  {cv_results[name]['cv_accuracy_mean']:.4f} (±{cv_results[name]['cv_accuracy_std']:.4f})"
        )

    return cv_results


def train_and_persist_models(
    X_train: np.ndarray, y_train: pd.Series, save_artifacts: bool = True
) -> dict:
    """Train all 3 classifiers on full training set and persist to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Compute positive weight for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos = float(neg_count / max(pos_count, 1))

    models = get_models(scale_pos_weight=scale_pos)

    # 1. Cross Validation on training split
    cv_metrics = evaluate_cross_validation(models, X_train, y_train)

    # Model Selection: Predefined criterion is highest mean 5-fold CV recall on training set
    selected_model_name = max(
        cv_metrics.keys(),
        key=lambda k: (cv_metrics[k]["cv_recall_mean"], cv_metrics[k]["cv_f1_mean"])
    )
    print("\n" + "=" * 65)
    print("MODEL SELECTION RESULT (5-Fold Stratified CV on Training Set):")
    print(f"Selected model based on mean 5-fold CV recall: {selected_model_name}")
    print(f"  Mean CV Recall:   {cv_metrics[selected_model_name]['cv_recall_mean']:.4f} (±{cv_metrics[selected_model_name]['cv_recall_std']:.4f})")
    print(f"  Mean CV ROC-AUC:  {cv_metrics[selected_model_name]['cv_roc_auc_mean']:.4f} (±{cv_metrics[selected_model_name]['cv_roc_auc_std']:.4f})")
    print(f"  Mean CV F1-Score: {cv_metrics[selected_model_name]['cv_f1_mean']:.4f} (±{cv_metrics[selected_model_name]['cv_f1_std']:.4f})")
    print("=" * 65 + "\n")

    # 2. Train on full X_train
    trained_models = {}
    print("--- Training Final Models on Full Training Split ---")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    # 3. Persist Models
    if save_artifacts:
        joblib.dump(trained_models["Logistic Regression"], LOGISTIC_REGRESSION_PATH)
        joblib.dump(trained_models["Random Forest"], RANDOM_FOREST_PATH)
        joblib.dump(trained_models["XGBoost"], XGBOOST_PATH)
        print(f"Serialized models saved to {MODELS_DIR}")

    return {
        "models": trained_models,
        "cv_metrics": cv_metrics,
        "selected_model": selected_model_name,
    }


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessor import prepare_and_split_data

    raw_df = load_raw_data()
    splits = prepare_and_split_data(raw_df)
    train_results = train_and_persist_models(splits["X_train_trans"], splits["y_train"])
    print("Training phase completed successfully.")
