"""
Evaluation & Comparison Module for Heart Disease Risk Prediction.

Computes precision, recall, F1-score classification reports,
generates side-by-side Confusion Matrices, and overlays combined ROC-AUC curve comparisons.
"""

import os
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate_models(
    rf_model: Any,
    ann_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "plots"
) -> Dict[str, Any]:
    """
    Evaluates tuned Random Forest and Keras ANN models on test data.
    Prints classification reports, plots confusion matrices, and ROC-AUC curve comparison.

    Parameters:
        rf_model: Fitted Random Forest model instance.
        ann_model: Fitted Keras Sequential ANN model instance.
        X_test (np.ndarray): Scaled test features.
        y_test (np.ndarray): Test labels.
        output_dir (str): Folder path to save output graphics.

    Returns:
        Dict[str, Any]: Evaluation metrics dictionary for both models.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n==========================================")
    print("        MODEL EVALUATION & COMPARISON     ")
    print("==========================================")

    # 1. Generate Predictions & Probabilities
    # Random Forest
    rf_pred_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = rf_model.predict(X_test)

    # ANN
    ann_pred_probs = ann_model.predict(X_test).ravel()
    ann_preds = (ann_pred_probs >= 0.5).astype(int)

    # 2. Print Classification Reports
    print("\n------------------------------------------")
    print("  MODEL A: RANDOM FOREST CLASSIFICATION REPORT")
    print("------------------------------------------")
    rf_report = classification_report(y_test, rf_preds, target_names=["Low Risk (0)", "High Risk (1)"])
    print(rf_report)

    print("\n------------------------------------------")
    print("  MODEL B: KERAS ANN CLASSIFICATION REPORT")
    print("------------------------------------------")
    ann_report = classification_report(y_test, ann_preds, target_names=["Low Risk (0)", "High Risk (1)"])
    print(ann_report)

    # 3. Calculate ROC-AUC Metrics
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_probs)
    rf_auc = auc(rf_fpr, rf_tpr)

    ann_fpr, ann_tpr, _ = roc_curve(y_test, ann_pred_probs)
    ann_auc = auc(ann_fpr, ann_tpr)

    print("\n--- Summary Performance Metrics ---")
    metrics_summary = {
        "Random Forest": {
            "Accuracy": accuracy_score(y_test, rf_preds),
            "Precision": precision_score(y_test, rf_preds),
            "Recall": recall_score(y_test, rf_preds),
            "F1-Score": f1_score(y_test, rf_preds),
            "ROC-AUC": rf_auc
        },
        "Keras ANN": {
            "Accuracy": accuracy_score(y_test, ann_preds),
            "Precision": precision_score(y_test, ann_preds),
            "Recall": recall_score(y_test, ann_preds),
            "F1-Score": f1_score(y_test, ann_preds),
            "ROC-AUC": ann_auc
        }
    }

    for model_name, m in metrics_summary.items():
        print(f"{model_name}:")
        print(f"  Accuracy:  {m['Accuracy']:.4f}")
        print(f"  Precision: {m['Precision']:.4f}")
        print(f"  Recall:    {m['Recall']:.4f}")
        print(f"  F1-Score:  {m['F1-Score']:.4f}")
        print(f"  ROC-AUC:   {m['ROC-AUC']:.4f}\n")

    # 4. Plot Confusion Matrices
    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    rf_cm = confusion_matrix(y_test, rf_preds)
    sns.heatmap(
        rf_cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        ax=axes[0], 
        cbar=False,
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"]
    )
    axes[0].set_title(f"Random Forest Confusion Matrix\nAccuracy: {metrics_summary['Random Forest']['Accuracy']:.2%}", fontweight="bold", fontsize=12)
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    ann_cm = confusion_matrix(y_test, ann_preds)
    sns.heatmap(
        ann_cm, 
        annot=True, 
        fmt="d", 
        cmap="Greens", 
        ax=axes[1], 
        cbar=False,
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"]
    )
    axes[1].set_title(f"Keras ANN Confusion Matrix\nAccuracy: {metrics_summary['Keras ANN']['Accuracy']:.2%}", fontweight="bold", fontsize=12)
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATION] Saved Confusion Matrices plot to '{cm_path}'")

    # 5. Plot Combined ROC-AUC Curve Comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(rf_fpr, rf_tpr, color="#2980b9", lw=2.5, label=f"Random Forest (AUC = {rf_auc:.3f})")
    ax.plot(ann_fpr, ann_tpr, color="#27ae60", lw=2.5, label=f"Keras ANN (AUC = {ann_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Classifier baseline")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title("ROC-AUC Comparison: Random Forest vs. Keras ANN", fontsize=14, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=11, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    roc_path = os.path.join(output_dir, "roc_auc_comparison.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATION] Saved ROC-AUC Comparison plot to '{roc_path}'")

    return metrics_summary
