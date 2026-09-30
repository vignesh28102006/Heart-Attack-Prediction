import streamlit as st
import pandas as pd
from src.database import get_prediction_history, clear_history


def render_history_view():
    st.header("📜 Clinical Prediction Audit History")
    st.markdown(
        """
        Historical record of all clinical risk predictions recorded in the relational database.
        Allows retrospective auditing and clinical cohort monitoring.
        """
    )

    records = get_prediction_history(limit=500)

    if not records:
        st.info("ℹ️ No patient risk assessments logged yet. Run a prediction to see audit entries here.")
        return

    df_history = pd.DataFrame(records)

    # Top Controls & Filters
    col_filter, col_actions = st.columns([3, 1])

    with col_filter:
        risk_options = ["All Records"] + list(df_history["risk_level"].unique())
        selected_risk = st.selectbox("Filter by Clinical Risk Level:", options=risk_options)

    with col_actions:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear History", help="Delete all audit log records"):
            clear_history()
            st.warning("Database records cleared.")
            st.rerun()

    # Filter dataset
    display_df = df_history.copy()
    if selected_risk != "All Records":
        display_df = display_df[display_df["risk_level"] == selected_risk]

    st.markdown(f"**Showing {len(display_df)} / {len(df_history)} recorded patient evaluations:**")

    # Reorder columns for clinical readability
    preferred_cols = [
        "id", "created_at", "prediction", "probability", "risk_level",
        "model_used", "age", "sex", "cp", "trestbps", "chol",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]
    present_cols = [c for c in preferred_cols if c in display_df.columns]
    
    st.dataframe(
        display_df[present_cols],
        use_container_width=True,
        height=380,
        hide_index=True,
    )

    # Export to CSV
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Export Audit Log to CSV",
        data=csv_bytes,
        file_name="cardiosense_prediction_audit.csv",
        mime="text/csv",
    )
