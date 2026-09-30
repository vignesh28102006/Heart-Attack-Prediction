import streamlit as st
import pandas as pd
from config import RAW_DATA_PATH, FEATURE_DESCRIPTIONS
from app.components.charts import render_metric_card


def render_overview_view():
    st.header("📋 UCI Heart Disease Dataset Overview")
    st.markdown(
        """
        The **Cleveland Heart Disease Database** from the **UCI Machine Learning Repository** 
        remains one of the most widely cited benchmarks for medical machine learning. 
        It records clinical measurements from patients undergoing coronary angiography at the 
        Cleveland Clinic Foundation.
        """
    )

    if not RAW_DATA_PATH.exists():
        st.warning("Dataset not found on disk. Run data pipeline to download.")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    # Metric Banner
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card("Total Patients", f"{len(df)}", "Cohort instances")
    with col2:
        render_metric_card("Clinical Features", f"{len(df.columns) - 1}", "Diagnostic predictors")
    with col3:
        disease_count = (df["target"] == 1).sum()
        disease_pct = (disease_count / len(df)) * 100
        render_metric_card("Disease Prevalence", f"{disease_pct:.1f}%", f"{disease_count} positive cases")
    with col4:
        missing_count = int(df.isnull().sum().sum())
        render_metric_card("Missing Values", f"{missing_count}", "Imputed with median/mode")

    st.markdown("---")

    # Data Explorer
    st.subheader("🔍 Patient Cohort Data Explorer")
    show_disease = st.selectbox(
        "Filter by Target Class:",
        options=["All Patients", "Heart Disease (1)", "No Disease (0)"],
    )

    filtered_df = df.copy()
    if show_disease == "Heart Disease (1)":
        filtered_df = filtered_df[filtered_df["target"] == 1]
    elif show_disease == "No Disease (0)":
        filtered_df = filtered_df[filtered_df["target"] == 0]

    st.dataframe(filtered_df, use_container_width=True, height=320)

    # Statistical Summary Tab
    with st.expander("📊 Numerical Descriptive Statistics", expanded=False):
        st.dataframe(df.describe().T, use_container_width=True)

    # Feature Dictionary Table
    st.subheader("📖 Clinical Attribute Dictionary")
    feature_table = []
    for col, desc in FEATURE_DESCRIPTIONS.items():
        feature_table.append({
            "Feature Name": col,
            "Clinical Description": desc,
            "Data Type": str(df[col].dtype) if col in df.columns else "N/A",
        })
    st.table(pd.DataFrame(feature_table))
