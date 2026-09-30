import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from config import EDA_PLOTS_DIR, RAW_DATA_PATH


def generate_all_eda_plots(df: pd.DataFrame, output_dir: Path = EDA_PLOTS_DIR):
    """Generate high-resolution EDA figures for Streamlit views and documentation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 10})

    # 1. Target Distribution
    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts = df["target"].value_counts().sort_index()
    labels = ["Healthy (0)", "Heart Disease (1)"]
    colors = ["#2b8cbe", "#e41a1c"]
    bars = ax.bar(labels, counts.values, color=colors, width=0.55, edgecolor="black", alpha=0.85)

    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.annotate(
            f"{h} ({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Prevalence of Heart Disease in Cleveland Cohort", fontsize=12, fontweight="bold")
    ax.set_ylabel("Patient Count")
    ax.set_ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=180)
    plt.close()

    # 2. Chest Pain Type vs Heart Disease Prevalence
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cp_labels = {
        0: "0: Typical Angina",
        1: "1: Atypical Angina",
        2: "2: Non-Anginal",
        3: "3: Asymptomatic",
    }
    df_cp = df.copy()
    df_cp["cp_label"] = df_cp["cp"].map(cp_labels)
    df_cp["diagnosis"] = df_cp["target"].map({0: "No Disease", 1: "Heart Disease"})

    sns.countplot(
        data=df_cp,
        x="cp_label",
        hue="diagnosis",
        palette=["#3182bd", "#de2d26"],
        ax=ax,
        edgecolor="black",
        alpha=0.9,
    )
    ax.set_title("Chest Pain Type Breakdown by Diagnosis", fontsize=12, fontweight="bold")
    ax.set_xlabel("Chest Pain Classification")
    ax.set_ylabel("Number of Patients")
    ax.legend(title="Diagnosis")
    plt.tight_layout()
    plt.savefig(output_dir / "chest_pain_distribution.png", dpi=180)
    plt.close()

    # 3. Age vs Cholesterol by Target
    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = sns.scatterplot(
        data=df,
        x="age",
        y="chol",
        hue="target",
        palette={0: "#2b8cbe", 1: "#e41a1c"},
        alpha=0.8,
        s=60,
        ax=ax,
        edgecolor="k",
    )
    # Add clinical reference lines
    ax.axhline(200, color="gray", linestyle="--", alpha=0.7, label="Borderline Chol (200 mg/dl)")
    ax.axvline(55, color="purple", linestyle=":", alpha=0.7, label="Age 55 Threshold")
    ax.set_title("Serum Cholesterol vs Age Segregated by Diagnosis", fontsize=12, fontweight="bold")
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Serum Cholesterol (mg/dl)")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, ["No Disease", "Heart Disease", "Chol 200 mg/dl", "Age 55"], loc="upper right")
    plt.tight_layout()
    plt.savefig(output_dir / "age_chol_by_target.png", dpi=180)
    plt.close()

    # 4. Maximum Heart Rate (thalach) Boxplots
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df_hr = df.copy()
    df_hr["diagnosis"] = df_hr["target"].map({0: "No Disease", 1: "Heart Disease"})
    sns.boxplot(
        data=df_hr,
        x="diagnosis",
        y="thalach",
        palette=["#74c476", "#fb6a4a"],
        width=0.45,
        ax=ax,
    )
    sns.stripplot(
        data=df_hr,
        x="diagnosis",
        y="thalach",
        color="black",
        alpha=0.3,
        jitter=0.2,
        size=4,
        ax=ax,
    )
    ax.set_title("Max Achieved Heart Rate (thalach) Distribution", fontsize=12, fontweight="bold")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Max Heart Rate (bpm)")
    plt.tight_layout()
    plt.savefig(output_dir / "max_hr_boxplots.png", dpi=180)
    plt.close()

    # 5. Correlation Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    numeric_df = df.apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-0.6,
        vmax=0.6,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap of Clinical Attributes", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=180)
    plt.close()

    print(f"All 5 EDA plots successfully written to {output_dir}")


if __name__ == "__main__":
    from src.data_loader import load_raw_data

    raw_data = load_raw_data()
    generate_all_eda_plots(raw_data)
