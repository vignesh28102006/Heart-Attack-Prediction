import streamlit as st
import pandas as pd
from pathlib import Path
from config import EDA_PLOTS_DIR, RAW_DATA_PATH


def render_eda_view():
    st.header("📊 Exploratory Clinical Data Analysis")
    st.markdown(
        """
        Deep clinical insights into the correlation structures, risk distributions, 
        and physiological predictors associated with ischemic heart disease.
        """
    )

    if not EDA_PLOTS_DIR.exists():
        st.warning("EDA plots not found. Please run `python -m src.eda_generator`.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Correlation Heatmap",
        "💔 Diagnosis & Chest Pain",
        "🫀 Exercise Stress & Max HR",
        "📈 Age vs Cholesterol",
    ])

    with tab1:
        st.subheader("Clinical Feature Correlation Matrix")
        st.markdown(
            """
            - **Positive Correlations with Disease:** `cp` (chest pain type), `oldpeak` (exercise ST depression), and `ca` (fluoroscopy vessels).
            - **Protective / Negative Correlations:** `thalach` (max heart rate achieved) shows a strong inverse correlation (-0.42), 
              as compromised cardiovascular systems exhibit chronotropic incompetence (inability to reach peak heart rate under stress).
            """
        )
        corr_img = EDA_PLOTS_DIR / "correlation_heatmap.png"
        if corr_img.exists():
            st.image(str(corr_img), use_container_width=True)

    with tab2:
        st.subheader("Disease Prevalence & Chest Pain Manifestations")
        col1, col2 = st.columns(2)
        with col1:
            target_img = EDA_PLOTS_DIR / "target_distribution.png"
            if target_img.exists():
                st.image(str(target_img), caption="Cohort Diagnosis Prevalence (54.1% Normal vs 45.9% Disease)", use_container_width=True)
        with col2:
            cp_img = EDA_PLOTS_DIR / "chest_pain_distribution.png"
            if cp_img.exists():
                st.image(str(cp_img), caption="Chest Pain Type vs Confirmed Disease", use_container_width=True)

        st.info(
            "💡 **Clinical Takeaway:** Counter-intuitively in this cohort, **asymptomatic chest pain (Type 3)** correlates with the highest proportion of angiographically confirmed CAD, underscoring the lethal nature of silent ischemia."
        )

    with tab3:
        st.subheader("Treadmill Stress Test: Maximum Heart Rate (thalach)")
        hr_img = EDA_PLOTS_DIR / "max_hr_boxplots.png"
        if hr_img.exists():
            st.image(str(hr_img), caption="Max Heart Rate Boxplot Distribution", use_container_width=True)
        st.markdown(
            """
            - Healthy patients regularly exceed **160 bpm** during standard Bruce protocol exercise stress testing.
            - Patients with severe coronary artery disease struggle to achieve target peak rates, frequently failing to cross **135 bpm**.
            """
        )

    with tab4:
        st.subheader("Serum Cholesterol vs Age Stratification")
        chol_img = EDA_PLOTS_DIR / "age_chol_by_target.png"
        if chol_img.exists():
            st.image(str(chol_img), caption="Cholesterol vs Age Segregated by Diagnosis", use_container_width=True)
        st.markdown(
            """
            - Hypercholesterolemia (> 240 mg/dl) is common across both cohorts, demonstrating that **cholesterol alone is not an independent differentiator** without stress test indicators (`oldpeak`, `thalach`, and `slope`).
            """
        )
