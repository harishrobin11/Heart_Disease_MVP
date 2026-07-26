"""
Model Evaluation Module for Heart Disease MLOps Architecture.

Prints detailed classification reports and saves ROC-AUC curve comparisons 
to 'artifacts/roc_auc_comparison.png'.
"""

import os
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score


def evaluate_models(
    rf_model: Any,
    ann_model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    artifact_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Evaluates trained Random Forest and Keras ANN models on the test set.
    Prints classification reports and saves ROC-AUC plot to '{artifact_dir}/roc_auc_comparison.png'.

    Parameters:
        rf_model: Trained Random Forest model instance.
        ann_model: Trained Keras Sequential model instance.
        X_test (np.ndarray): Scaled test feature matrix.
        y_test (np.ndarray): Unaltered test target labels.
        artifact_dir (str): Directory path to save ROC plot artifact.

    Returns:
        Dict[str, Any]: Dictionary containing ROC-AUC and accuracy summary metrics.
    """
    os.makedirs(artifact_dir, exist_ok=True)

    print("\n==========================================")
    print("      EVALUATING MODEL PERFORMANCE        ")
    print("==========================================")

    # 1. Predictions & Probabilities
    rf_pred_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_preds = rf_model.predict(X_test)

    ann_pred_probs = ann_model.predict(X_test, verbose=0).ravel()
    ann_preds = (ann_pred_probs >= 0.5).astype(int)

    # 2. Print Classification Reports
    print("\n------------------------------------------")
    print(" MODEL A: RANDOM FOREST CLASSIFICATION REPORT")
    print("------------------------------------------")
    print(classification_report(y_test, rf_preds, target_names=["Low Risk (0)", "High Risk (1)"]))

    print("------------------------------------------")
    print(" MODEL B: KERAS ANN CLASSIFICATION REPORT")
    print("------------------------------------------")
    print(classification_report(y_test, ann_preds, target_names=["Low Risk (0)", "High Risk (1)"]))

    # 3. Compute ROC-AUC Scores
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_probs)
    rf_auc = auc(rf_fpr, rf_tpr)

    ann_fpr, ann_tpr, _ = roc_curve(y_test, ann_pred_probs)
    ann_auc = auc(ann_fpr, ann_tpr)

    print("--- Summary Performance Metrics ---")
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_preds):.4f} | ROC-AUC: {rf_auc:.4f}")
    print(f"Keras ANN     Accuracy: {accuracy_score(y_test, ann_preds):.4f} | ROC-AUC: {ann_auc:.4f}")

    # 4. Save Overlaid ROC-AUC Comparison Plot
    plt.figure(figsize=(8, 6))
    plt.plot(rf_fpr, rf_tpr, color="#2980b9", lw=2.5, label=f"Random Forest (AUC = {rf_auc:.3f})")
    plt.plot(ann_fpr, ann_tpr, color="#27ae60", lw=2.5, label=f"Keras ANN (AUC = {ann_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="#7f8c8d", lw=1.5, linestyle="--", label="Random Baseline")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    plt.title("ROC-AUC Curve Comparison: Random Forest vs. Keras ANN", fontsize=13, fontweight="bold", pad=12)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)

    roc_path = os.path.join(artifact_dir, "roc_auc_comparison.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[EVALUATE] Saved ROC-AUC Comparison plot to '{roc_path}'")

    return {
        "random_forest_auc": rf_auc,
        "ann_auc": ann_auc,
        "random_forest_accuracy": accuracy_score(y_test, rf_preds),
        "ann_accuracy": accuracy_score(y_test, ann_preds)
    }
