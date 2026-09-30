import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def render_metric_card(label: str, value: str, subtext: str = "", delta_color: str = "normal"):
    """Render a styled metric badge."""
    st.markdown(
        f"""
        <div style="background: rgba(255, 255, 255, 0.05); padding: 18px; border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 12px;">
            <div style="color: #94a3b8; font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;">{label}</div>
            <div style="color: #38bdf8; font-size: 1.9rem; font-weight: 700; margin-top: 4px;">{value}</div>
            <div style="color: #64748b; font-size: 0.8rem; margin-top: 2px;">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_gauge(probability: float):
    """Render an aesthetic gauge/progress card for heart disease risk."""
    pct = probability * 100
    if probability < 0.35:
        color = "#10b981"  # Emerald green
        badge_text = "LOW RISK"
        desc = "Model prediction suggests lower probability of heart disease based on input features."
    elif probability <= 0.65:
        color = "#f59e0b"  # Amber
        badge_text = "MODERATE RISK"
        desc = "Model prediction suggests intermediate probability based on input features."
    else:
        color = "#ef4444"  # Rose red
        badge_text = "HIGH RISK"
        desc = "Model prediction suggests elevated probability of heart disease based on input features."

    st.markdown(
        f"""
        <div style="background: #1e293b; padding: 24px; border-radius: 12px; border: 1px solid {color}40; margin: 15px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">Predicted Risk Probability</span>
                <span style="background: {color}20; color: {color}; border: 1px solid {color}; padding: 4px 12px; border-radius: 20px; font-weight: 700; font-size: 0.85rem;">
                    {badge_text}
                </span>
            </div>
            <div style="font-size: 2.8rem; font-weight: 800; color: {color}; margin: 8px 0;">
                {pct:.1f}%
            </div>
            <div style="width: 100%; background: #334155; border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 12px;">
                <div style="width: {pct}%; background: {color}; height: 100%; border-radius: 8px; transition: width 0.5s ease;"></div>
            </div>
            <div style="color: #cbd5e1; font-size: 0.9rem;">
                {desc}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
