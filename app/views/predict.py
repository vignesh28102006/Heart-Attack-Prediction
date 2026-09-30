import joblib
import pandas as pd
import streamlit as st

from app.components.charts import render_risk_gauge
from app.components.forms import render_patient_form
from config import (
    ALL_FEATURES,
    LOGISTIC_REGRESSION_PATH,
    METRICS_PATH,
    PREPROCESSOR_PATH,
    RANDOM_FOREST_PATH,
    XGBOOST_PATH,
)
from src.database import save_prediction


@st.cache_resource
def load_ml_pipeline():
    """Load serialized preprocessor and models into Streamlit cache."""
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    models = {
        "XGBoost (Highest Test Recall: 96.43%)": joblib.load(XGBOOST_PATH),
        "Random Forest Classifier": joblib.load(RANDOM_FOREST_PATH),
        "Logistic Regression (Baseline)": joblib.load(LOGISTIC_REGRESSION_PATH),
    }
    return preprocessor, models


def render_predict_view():
    st.header("🩺 Patient Risk Prediction")
    st.markdown(
        """
        Estimate cardiovascular disease risk prediction by inputting patient indicators.
        Predictions are logged to the database for history tracking and auditing.
        """
    )

    try:
        preprocessor, models = load_ml_pipeline()
    except Exception as e:
        st.error(f"Error loading machine learning models: {e}. Please ensure models are trained.")
        return

    # Model Selector
    model_choice = st.selectbox(
        "Select Machine Learning Engine:",
        options=list(models.keys()),
        index=0,
        help="XGBoost demonstrated the highest recall on the held-out test set (0.9643).",
    )
    selected_model = models[model_choice]

    # Render Patient Form
    patient_data = render_patient_form()

    # Predict Button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 Run Risk Prediction", type="primary", use_container_width=True):
        input_df = pd.DataFrame([patient_data])
        input_df = input_df[ALL_FEATURES]

        try:
            # Transform features through preprocessor
            transformed_features = preprocessor.transform(input_df)

            # Predict class & probability
            prediction = int(selected_model.predict(transformed_features)[0])
            if hasattr(selected_model, "predict_proba"):
                prob = float(selected_model.predict_proba(transformed_features)[0, 1])
            else:
                prob = float(prediction)

            st.markdown("### 📋 Prediction Outcome")
            render_risk_gauge(prob)

            # Persist to Database
            clean_model_name = model_choice.split(" (")[0]
            record = save_prediction(
                feature_dict=patient_data,
                model_name=clean_model_name,
                prediction=prediction,
                probability=prob,
            )

            st.success(
                f"✅ **Prediction Record #{record.id} successfully saved to Audit Database.** "
                f"Model: {clean_model_name} | Assigned Risk Category: **{record.risk_level}**"
            )

            # Key indicators summary
            with st.expander("🔬 Input Feature Summary", expanded=False):
                st.json(patient_data)

        except Exception as e:
            st.error(f"Inference execution failed: {e}")
