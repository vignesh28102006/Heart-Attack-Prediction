import json
import streamlit as st
import pandas as pd
from config import METRICS_PATH, MODELS_DIR
from app.components.charts import render_metric_card


def render_model_comparison_view():
    st.header("⚖️ Model Selection & Final Evaluation")
    st.markdown(
        """
        Methodological comparison of **Logistic Regression (Baseline)**, **Random Forest**, and **XGBoost Classifier**.
        This page clearly distinguishes **Model Selection** (via 5-Fold CV on training data) from the **Final Held-Out Test Evaluation**.
        """
    )

    if not METRICS_PATH.exists():
        st.error("Evaluation metrics not found. Please run `python -m src.evaluate`.")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    models_dict = metrics_data["models"]
    selected_model = metrics_data.get("selected_model", metrics_data.get("best_model", "Logistic Regression"))

    # =========================================================================
    # PART 1: MODEL SELECTION RESULT (TRAINING DATA 5-FOLD CV)
    # =========================================================================
    st.markdown("---")
    st.subheader("1. 🎯 Model Selection Result (5-Fold Stratified Cross-Validation)")
    st.markdown(
        """
        To prevent model-selection bias, **model selection was performed strictly on the 242-sample training set**
        using 5-fold Stratified Cross-Validation. The holdout test set was **not** used to choose the model.
        """
    )

    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); padding: 18px; border-radius: 10px; border-left: 6px solid #38bdf8; margin-bottom: 16px;">
            <div style="color: #38bdf8; font-weight: 700; font-size: 1.05rem;">
                🏆 Selected Model Based on Mean 5-Fold CV Recall: <u>{selected_model}</u>
            </div>
            <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 6px;">
                <b>Selection Criterion:</b> Highest Mean Recall across 5 Stratified Folds on training data.
                Logistic Regression demonstrated the highest generalizable sensitivity on the training set (80.99% ± 6.16%).
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Cross-Validation Comparison Table
    cv_table = []
    for name, m in models_dict.items():
        cv = m.get("cv_metrics", {})
        if cv:
            cv_table.append({
                "Model": name,
                "Selection Status": "✅ Selected Model" if name == selected_model else "Evaluated Baseline",
                "Mean CV Recall (Primary) ⭐": f"{cv.get('cv_recall_mean', 0.0):.4f} (±{cv.get('cv_recall_std', 0.0):.4f})",
                "Mean CV ROC-AUC": f"{cv.get('cv_roc_auc_mean', 0.0):.4f} (±{cv.get('cv_roc_auc_std', 0.0):.4f})",
                "Mean CV F1-Score": f"{cv.get('cv_f1_mean', 0.0):.4f} (±{cv.get('cv_f1_std', 0.0):.4f})",
                "Mean CV Accuracy": f"{cv.get('cv_accuracy_mean', 0.0):.4f} (±{cv.get('cv_accuracy_std', 0.0):.4f})",
            })
    if cv_table:
        st.dataframe(pd.DataFrame(cv_table), use_container_width=True, hide_index=True)

    # =========================================================================
    # PART 2: FINAL HELD-OUT TEST EVALUATION (UNTOUCHED TEST SET)
    # =========================================================================
    st.markdown("---")
    st.subheader("2. 🧪 Final Held-Out Test Evaluation (Untouched Test Set — 61 Patients)")
    st.markdown(
        """
        After the model was selected on the training data, **all three models were evaluated once on the previously untouched 20% holdout test set**
        (33 negative, 28 positive cases) to report unbiased generalization performance.
        """
    )

    # Metric Cards for Top 3 on Test Set
    col1, col2, col3 = st.columns(3)
    for i, (name, m) in enumerate(models_dict.items()):
        target_col = [col1, col2, col3][i]
        with target_col:
            st.markdown(f"#### {name}")
            render_metric_card(
                "Test Recall",
                f"{m['recall']:.2%}",
                f"FN: {m['confusion_matrix']['fn']} / TP: {m['confusion_matrix']['tp']}",
            )
            render_metric_card("Test ROC-AUC", f"{m['roc_auc']:.3f}", "Test discrimination")
            render_metric_card("Test F1-Score", f"{m['f1_score']:.2%}", "Test harmonic mean")

    # Comparative Metrics Table on Test Set
    st.markdown("##### 📋 Complete Holdout Test Set Performance Table")
    table_data = []
    for name, m in models_dict.items():
        cm = m["confusion_matrix"]
        table_data.append({
            "Model": name,
            "Test Recall": f"{m['recall']:.4f}",
            "Test F1-Score": f"{m['f1_score']:.4f}",
            "Test ROC-AUC": f"{m['roc_auc']:.4f}",
            "Test Accuracy": f"{m['accuracy']:.4f}",
            "Test Precision": f"{m['precision']:.4f}",
            "False Negatives (FN)": cm["fn"],
            "True Positives (TP)": cm["tp"],
            "False Positives (FP)": cm["fp"],
            "True Negatives (TN)": cm["tn"],
        })
    df_metrics = pd.DataFrame(table_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Diagnostic Visualizations
    st.markdown("##### 🔬 Final Holdout Evaluation Artifacts")
    tab_cm, tab_roc, tab_bar = st.tabs(["Confusion Matrices", "ROC Curves", "Performance Bar Chart"])

    with tab_cm:
        cm_path = MODELS_DIR / "confusion_matrices.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Holdout Test Set Confusion Matrices (61 Patients)", use_container_width=True)

    with tab_roc:
        roc_path = MODELS_DIR / "roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)

    with tab_bar:
        bar_path = MODELS_DIR / "metrics_comparison.png"
        if bar_path.exists():
            st.image(str(bar_path), caption="Cross-Model Metric Benchmark on Test Set", use_container_width=True)
