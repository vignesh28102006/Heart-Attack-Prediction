import streamlit as st
import sys
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database import init_db
from app.views.overview import render_overview_view
from app.views.eda import render_eda_view
from app.views.model_comparison import render_model_comparison_view
from app.views.predict import render_predict_view
from app.views.history import render_history_view

# Streamlit Page Setup
st.set_page_config(
    page_title="CardioSense | Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Medical Aesthetic Styling
st.markdown(
    """
    <style>
    /* Global Styling */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #f8fafc;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #0284c7 0%, #0369a1 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #075985 100%);
        box-shadow: 0 4px 12px rgba(2, 132, 199, 0.3);
        transform: translateY(-1px);
    }
    
    /* Metric Cards & DataFrames */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #1e293b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize database
try:
    init_db()
except Exception as e:
    st.sidebar.warning(f"Database note: {e}")

# Sidebar Brand & Navigation
st.sidebar.markdown(
    """
    <div style="text-align: center; padding: 10px 0 20px 0;">
        <span style="font-size: 2.2rem;">❤️</span>
        <h2 style="margin: 0; color: #38bdf8; font-size: 1.4rem;">CardioSense ML</h2>
        <span style="font-size: 0.8rem; color: #94a3b8;">Clinical Heart Disease Prediction</span>
    </div>
    """,
    unsafe_allow_html=True,
)

NAV_OPTIONS = {
    "🩺 Patient Risk Predictor": render_predict_view,
    "⚖️ Model & Recall Comparison": render_model_comparison_view,
    "📊 Exploratory Data Analysis": render_eda_view,
    "📋 Dataset Overview": render_overview_view,
    "📜 Prediction Audit History": render_history_view,
}

selected_nav = st.sidebar.radio(
    "Navigation Menu",
    options=list(NAV_OPTIONS.keys()),
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="padding: 10px; font-size: 0.8rem; color: #64748b;">
        <b>Stack:</b> Scikit-learn, XGBoost, Streamlit, SQLAlchemy, PostgreSQL.<br><br>
        <b>Primary Metric:</b> Recall (identifying positive heart disease cases).
    </div>
    """,
    unsafe_allow_html=True,
)

# Mandatory Educational Disclaimer Banner
st.warning(
    "⚠️ **Educational & Research Notice:** This is an educational machine-learning prediction system "
    "and is not a medical diagnosis or a substitute for professional medical advice."
)

# Render Selected View
view_func = NAV_OPTIONS[selected_nav]
view_func()
