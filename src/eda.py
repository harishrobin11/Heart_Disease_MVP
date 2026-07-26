"""
Exploratory Data Analysis (EDA) Module for Heart Disease Risk Prediction.

Provides automated statistical summary, null value checking, feature correlation analysis,
and visualization generation for clinical distributions.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def perform_eda(df: pd.DataFrame, output_dir: str = "plots") -> None:
    """
    Performs exploratory data analysis, prints statistical summaries, checks for nulls,
    and generates visual plots saved to the specified directory.

    Parameters:
        df (pd.DataFrame): Input heart disease dataset.
        output_dir (str): Folder path to save visual plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n==========================================")
    print("       EXPLORATORY DATA ANALYSIS          ")
    print("==========================================")

    print("\n--- Dataset Shape ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    print("\n--- Dataset Data Types & Non-Null Counts ---")
    print(df.info())

    print("\n--- Null Value Count Per Feature ---")
    null_counts = df.isnull().sum()
    print(null_counts)
    if null_counts.sum() == 0:
        print(">> Clean dataset! No missing/null values detected.")

    print("\n--- Descriptive Statistics ---")
    print(df.describe().T.round(2))

    print("\n--- Target Class Distribution ---")
    target_counts = df['target'].value_counts()
    target_pcts = df['target'].value_counts(normalize=True) * 100
    for cls in target_counts.index:
        print(f"Class {cls}: {target_counts[cls]} patients ({target_pcts[cls]:.2f}%)")

    # Set aesthetics for plots
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

    # Plot 1: Target Class Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(
        data=df, 
        x="target", 
        hue="target",
        palette=["#2ecc71", "#e74c3c"], 
        legend=False,
        ax=ax
    )
    ax.set_title("Heart Disease Risk Target Distribution", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Target (0 = Low Risk / Healthy, 1 = High Risk / Disease)", fontsize=11)
    ax.set_ylabel("Patient Count", fontsize=11)
    ax.set_xticklabels(["0 (Low Risk)", "1 (High Risk)"])
    
    # Annotate bars with counts
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(f"{height}", (p.get_x() + p.get_width() / 2., height / 2),
                    ha="center", va="center", fontsize=12, color="white", fontweight="bold")

    target_plot_path = os.path.join(output_dir, "target_distribution.png")
    plt.savefig(target_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n[EDA] Saved Target Distribution plot to '{target_plot_path}'")

    # Plot 2: Correlation Heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr()
    mask = sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Clinical Features Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
    
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved Correlation Heatmap to '{heatmap_path}'")
