import json
from datetime import datetime, timezone
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    LOGISTIC_REGRESSION_PATH,
    METRICS_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    NUMERICAL_FEATURES,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
)


def evaluate_all_models(
    models: dict, X_test: np.ndarray, y_test: pd.Series, cv_results: dict = None
) -> dict:
    """Compute test metrics, generate diagnostic plots, and compile metrics JSON."""
    results = {}
    roc_data = {}

    print("\n================ TEST SET EVALUATION ================")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = y_pred

        acc = float(accuracy_score(y_test, y_pred))
        rec = float(recall_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_data[name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc}

        results[name] = {
            "recall": rec,  # Primary metric
            "accuracy": acc,
            "precision": prec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": {
                "matrix": cm,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "cv_metrics": cv_results.get(name, {}) if cv_results else {},
        }

        print(f"\nModel: {name}")
        print(f"  Recall (Sensitivity): {rec:.4f}  <-- PRIMARY METRIC")
        print(f"  ROC-AUC:              {auc:.4f}")
        print(f"  F1-Score:             {f1:.4f}")
        print(f"  Accuracy:             {acc:.4f}")
        print(f"  Precision:            {prec:.4f}")
        print(f"  Confusion Matrix:     TP={tp}, FP={fp}, TN={tn}, FN={fn}")

    # -------------------------------------------------------------------------
    # MODEL SELECTION:
    # Strictly selected using highest mean 5-fold CV recall on the training set.
    # The holdout test set is NEVER used for model selection or tuning.
    # -------------------------------------------------------------------------
    if cv_results:
        selected_model_name = max(
            cv_results.keys(),
            key=lambda k: (cv_results[k]["cv_recall_mean"], cv_results[k]["cv_f1_mean"])
        )
    else:
        # Fallback to standard baseline if CV not supplied
        selected_model_name = "Logistic Regression"

    print("\n" + "=" * 70)
    print("MODEL SELECTION RESULT (From Training Data 5-Fold Stratified CV):")
    print(f"Selected Model based on mean 5-fold CV recall: {selected_model_name}")
    if cv_results and selected_model_name in cv_results:
        sm_cv = cv_results[selected_model_name]
        print(f"  Mean CV Recall:    {sm_cv['cv_recall_mean']:.4f} (±{sm_cv['cv_recall_std']:.4f})  <-- SELECTION CRITERION")
        print(f"  Mean CV ROC-AUC:   {sm_cv['cv_roc_auc_mean']:.4f} (±{sm_cv['cv_roc_auc_std']:.4f})")
        print(f"  Mean CV F1-Score:  {sm_cv['cv_f1_mean']:.4f} (±{sm_cv['cv_f1_std']:.4f})")
        print(f"  Mean CV Accuracy:  {sm_cv['cv_accuracy_mean']:.4f} (±{sm_cv['cv_accuracy_std']:.4f})")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("FINAL HELD-OUT TEST EVALUATION (Untouched 20% Holdout Test Set - 61 Patients):")
    for name, m in results.items():
        print(f"[{name}] Test Recall: {m['recall']:.4f} | Test ROC-AUC: {m['roc_auc']:.4f} | Test F1: {m['f1_score']:.4f} | Test Acc: {m['accuracy']:.4f}")
    print("=" * 70)

    # Save metrics JSON with explicit methodological separation
    output_payload = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_selection": {
            "methodology": "5-Fold Stratified Cross-Validation on 242-sample training set",
            "primary_metric": "mean_cv_recall",
            "selected_model": selected_model_name,
            "cv_metrics_summary": cv_results if cv_results else {},
        },
        "final_held_out_test_evaluation": {
            "test_sample_count": len(y_test),
            "test_class_distribution": {
                "0": int((y_test == 0).sum()),
                "1": int((y_test == 1).sum()),
            },
            "unseen_holdout_status": "Untouched during preprocessing and model selection",
        },
        "selected_model": selected_model_name,
        "best_model": selected_model_name,  # Backward compatibility for legacy readers
        "models": results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # Save model metadata
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_selection_basis": "Mean 5-Fold Stratified Cross-Validation Recall on Training Data",
        "selected_model": selected_model_name,
        "best_model": selected_model_name,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": ALL_FEATURES,
        "target_mapping": {"0": "No Disease", "1": "Heart Disease"},
        "models_available": list(models.keys()),
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Generate and save diagnostic figures
    generate_evaluation_plots(results, roc_data)

    return output_payload


def generate_evaluation_plots(results: dict, roc_data: dict):
    """Render high-resolution figures for confusion matrices, ROC curves, and metric comparison."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrices Subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for i, (name, metrics) in enumerate(results.items()):
        cm = np.array(metrics["confusion_matrix"]["matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues" if i != 2 else "crest",
            cbar=False,
            ax=axes[i],
            annot_kws={"size": 14, "weight": "bold"},
            xticklabels=["No Disease (0)", "Disease (1)"],
            yticklabels=["No Disease (0)", "Disease (1)"],
        )
        axes[i].set_title(
            f"{name}\nRecall: {metrics['recall']:.2%} | F1: {metrics['f1_score']:.2%}",
            fontsize=12,
            fontweight="bold",
        )
        axes[i].set_xlabel("Predicted Class", fontsize=11)
        axes[i].set_ylabel("Actual Class", fontsize=11)

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "confusion_matrices.png", dpi=200)
    plt.close()
    print(f"Saved: {MODELS_DIR / 'confusion_matrices.png'}")

    # 2. Combined ROC Curves
    plt.figure(figsize=(8, 6))
    colors = ["#2b5c8f", "#d95f02", "#1b9e77"]
    for i, (name, r_info) in enumerate(roc_data.items()):
        plt.plot(
            r_info["fpr"],
            r_info["tpr"],
            color=colors[i % len(colors)],
            lw=2.5,
            label=f"{name} (AUC = {r_info['auc']:.3f})",
        )
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.500)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
    plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "roc_curves.png", dpi=200)
    plt.close()
    print(f"Saved: {MODELS_DIR / 'roc_curves.png'}")

    # 3. Bar Chart of Metrics Across Models
    metric_names = ["Recall", "Accuracy", "Precision", "F1-Score", "ROC-AUC"]
    plot_rows = []
    for model_name, m in results.items():
        plot_rows.append({"Model": model_name, "Metric": "Recall (Primary)", "Score": m["recall"]})
        plot_rows.append({"Model": model_name, "Metric": "Accuracy", "Score": m["accuracy"]})
        plot_rows.append({"Model": model_name, "Metric": "Precision", "Score": m["precision"]})
        plot_rows.append({"Model": model_name, "Metric": "F1-Score", "Score": m["f1_score"]})
        plot_rows.append({"Model": model_name, "Metric": "ROC-AUC", "Score": m["roc_auc"]})

    df_plot = pd.DataFrame(plot_rows)

    plt.figure(figsize=(10, 5.5))
    ax = sns.barplot(
        data=df_plot,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#3182bd", "#31a354", "#de2d26"],
    )
    plt.title("Comparative Performance Across Classifiers", fontsize=13, fontweight="bold")
    plt.ylim([0.0, 1.05])
    plt.ylabel("Metric Score", fontsize=11)
    plt.xlabel("")
    plt.legend(loc="lower right")

    # Add numeric labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="bottom",
                fontsize=8,
                xytext=(0, 2),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(MODELS_DIR / "metrics_comparison.png", dpi=200)
    plt.close()
    print(f"Saved: {MODELS_DIR / 'metrics_comparison.png'}")


def load_all_models() -> dict:
    """Load serialized models from disk."""
    return {
        "Logistic Regression": joblib.load(LOGISTIC_REGRESSION_PATH),
        "Random Forest": joblib.load(RANDOM_FOREST_PATH),
        "XGBoost": joblib.load(XGBOOST_PATH),
    }


if __name__ == "__main__":
    from src.data_loader import load_raw_data
    from src.preprocessor import prepare_and_split_data
    from src.train import train_and_persist_models

    raw_df = load_raw_data()
    splits = prepare_and_split_data(raw_df)
    train_res = train_and_persist_models(splits["X_train_trans"], splits["y_train"])
    eval_res = evaluate_all_models(
        train_res["models"],
        splits["X_test_trans"],
        splits["y_test"],
        cv_results=train_res["cv_metrics"],
    )
    print("Full Evaluation Pipeline Complete.")
