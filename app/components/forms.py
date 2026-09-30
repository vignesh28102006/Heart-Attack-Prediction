import streamlit as st


def render_patient_form() -> dict:
    """Render structured two-column patient clinical assessment form."""
    st.markdown("### 📋 Enter Patient Diagnostic Metrics")
    st.markdown(
        "Provide patient clinical parameters obtained from baseline checkup, resting vitals, and treadmill stress ECG."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 👤 Baseline & Resting Vitals")
        age = st.slider("Age (years)", min_value=20, max_value=90, value=55, step=1)
        sex = st.selectbox(
            "Biological Sex",
            options=[1, 0],
            format_func=lambda x: "Male" if x == 1 else "Female",
        )
        cp = st.selectbox(
            "Chest Pain Type (cp)",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0: Typical Angina",
                1: "1: Atypical Angina",
                2: "2: Non-Anginal Pain",
                3: "3: Asymptomatic",
            }[x],
            index=0,
            help="Typical angina has the highest ischemic specificity.",
        )
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=240,
            value=130,
            step=2,
            help="Resting blood pressure on hospital admission.",
        )
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=245,
            step=5,
            help="Normal: < 200 mg/dl; Borderline: 200-239; High: ≥ 240",
        )
        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            options=[0, 1],
            format_func=lambda x: "True (> 120 mg/dl)" if x == 1 else "False (≤ 120 mg/dl)",
            index=0,
        )
        restecg = st.selectbox(
            "Resting Electrocardiographic Results",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0: Normal",
                1: "1: ST-T Wave Abnormality (T-inversions/elevations)",
                2: "2: Left Ventricular Hypertrophy (Estes' criteria)",
            }[x],
            index=1,
        )

    with col2:
        st.markdown("##### 🏃 Exercise Stress & Fluoroscopy")
        thalach = st.slider(
            "Maximum Heart Rate Achieved (thalach)",
            min_value=60,
            max_value=220,
            value=150,
            step=1,
            help="Highest beats per minute recorded during exercise stress test.",
        )
        exang = st.selectbox(
            "Exercise-Induced Angina (exang)",
            options=[0, 1],
            format_func=lambda x: "Yes (Angina provoked by exertion)" if x == 1 else "No",
            index=0,
        )
        oldpeak = st.slider(
            "ST Depression Induced by Exercise (oldpeak)",
            min_value=0.0,
            max_value=6.2,
            value=1.0,
            step=0.1,
            help="ST depression relative to baseline resting ECG.",
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: {
                0: "0: Upsloping",
                1: "1: Flat",
                2: "2: Downsloping",
            }[x],
            index=1,
        )
        ca = st.selectbox(
            "Major Vessels Colored by Fluoroscopy (ca)",
            options=[0, 1, 2, 3],
            index=0,
            help="Number of major coronary vessels (0-3) with visible stenosis.",
        )
        thal = st.selectbox(
            "Thallium Scintigraphy Result (thal)",
            options=[1, 2, 3],
            format_func=lambda x: {
                1: "1: Normal Perfusion",
                2: "2: Fixed Defect (Previous Myocardial Infarction)",
                3: "3: Reversible Defect (Active Ischemia)",
            }[x],
            index=1,
        )

    return {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }
