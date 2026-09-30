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

    # Identify model with highest recall on the held-out test set
    best_model_name = max(
        results.keys(), key=lambda k: (results[k]["recall"], results[k]["f1_score"])
    )
    print(
        f"\n>>> Model with highest recall on the held-out test set: {best_model_name} "
        f"(Recall: {results[best_model_name]['recall']:.4f}) <<<"
    )

    # Save metrics JSON
    output_payload = {
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "recall",
        "best_model": best_model_name,
        "models": results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"Metrics saved to {METRICS_PATH}")

    # Save model metadata
    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "recall",
        "best_model": best_model_name,
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
        axes[i].set_xlabel("Predicted Diagnosis", fontsize=11)
        axes[i].set_ylabel("Actual Diagnosis", fontsize=11)

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
