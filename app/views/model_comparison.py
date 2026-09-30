import json
import streamlit as st
import pandas as pd
from config import METRICS_PATH, MODELS_DIR
from app.components.charts import render_metric_card


def render_model_comparison_view():
    st.header("⚖️ Machine Learning Model & Recall Comparison")
    st.markdown(
        """
        Comparison of **Logistic Regression (Baseline)**, **Random Forest**, and **XGBoost Classifier**
        evaluated on the unseen test split (61 patients).
        """
    )

    if not METRICS_PATH.exists():
        st.error("Evaluation metrics not found. Please run `python -m src.evaluate`.")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)

    models_dict = metrics_data["models"]
    best_model = metrics_data.get("best_model", "XGBoost")

    # Primary Metric Callout
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%); padding: 20px; border-radius: 12px; border-left: 6px solid #38bdf8; margin-bottom: 24px;">
            <h4 style="margin: 0; color: #38bdf8; font-weight: 700;">🎯 Primary Evaluation Metric: RECALL (Sensitivity)</h4>
            <p style="margin-top: 8px; color: #e2e8f0; line-height: 1.5; font-size: 0.95rem;">
                In screening models, a <b>False Negative</b> (predicting a heart disease patient as healthy) is more critical than a False Positive. 
                Therefore, the primary objective is maximizing <b>Recall = TP / (TP + FN)</b> to minimize missed positive cases.
            </p>
            <div style="margin-top: 12px; font-weight: 600; color: #facc15;">
                📊 Model with highest recall on the held-out test set: <u>{best_model}</u> ({models_dict[best_model]['recall']:.2%} Recall, {models_dict[best_model]['confusion_matrix']['fn']} false negative out of {models_dict[best_model]['confusion_matrix']['tp'] + models_dict[best_model]['confusion_matrix']['fn']} positive test cases).
            </div>
            <div style="margin-top: 6px; font-size: 0.8rem; color: #94a3b8;">
                Methodological note: Test set scores are evaluated on 61 unseen holdout samples. This system is for educational/experimental benchmarking and is not clinically validated.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metric Cards for Top 3
    col1, col2, col3 = st.columns(3)
    for i, (name, m) in enumerate(models_dict.items()):
        target_col = [col1, col2, col3][i]
        with target_col:
            st.markdown(f"#### {name}")
            render_metric_card(
                "Recall (Primary)",
                f"{m['recall']:.2%}",
                f"FN: {m['confusion_matrix']['fn']} / TP: {m['confusion_matrix']['tp']}",
            )
            render_metric_card("ROC-AUC Score", f"{m['roc_auc']:.3f}", "Discrimination power")
            render_metric_card("F1-Score", f"{m['f1_score']:.2%}", "Harmonic mean")

    st.markdown("---")

    # Comparative Metrics Table
    st.subheader("📋 Comprehensive Test Set Performance Table")
    table_data = []
    for name, m in models_dict.items():
        cm = m["confusion_matrix"]
        table_data.append({
            "Model": name,
            "Recall (Primary) ⭐": f"{m['recall']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "Accuracy": f"{m['accuracy']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "False Negatives (FN)": cm["fn"],
            "True Positives (TP)": cm["tp"],
            "False Positives (FP)": cm["fp"],
            "True Negatives (TN)": cm["tn"],
        })
    df_metrics = pd.DataFrame(table_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)

    # Diagnostic Visualizations
    st.subheader("🔬 Clinical Diagnostic Artifacts")
    tab_cm, tab_roc, tab_bar = st.tabs(["Confusion Matrices", "ROC Curves", "Performance Bar Chart"])

    with tab_cm:
        cm_path = MODELS_DIR / "confusion_matrices.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Holdout Test Set Confusion Matrices (61 Patients)", use_container_width=True)

    with tab_roc:
        roc_path = MODELS_DIR / "roc_curves.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curves and Area Under Curve (AUC)", use_container_width=True)

    with tab_bar:
        bar_path = MODELS_DIR / "metrics_comparison.png"
        if bar_path.exists():
            st.image(str(bar_path), caption="Cross-Model Metric Benchmark", use_container_width=True)
