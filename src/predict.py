"""
Single Patient Prediction Module for Heart Disease Risk Assessment.

Provides functions to format patient clinical profiles, scale continuous features,
and run inference using trained Random Forest and Keras ANN models.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


# Clinical feature metadata and acceptable defaults
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", 
    "ca", "thal"
]
CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def predict_patient_risk(
    patient_data: Dict[str, Any],
    rf_model: Any,
    ann_model: Any,
    scaler: Any
) -> Dict[str, Any]:
    """
    Predicts heart disease risk for a single patient profile using trained models.

    Parameters:
        patient_data (Dict[str, Any]): Dictionary containing patient clinical metrics.
        rf_model: Trained Random Forest Classifier.
        ann_model: Trained Keras Sequential ANN.
        scaler: Fitted StandardScaler instance.

    Returns:
        Dict[str, Any]: Comprehensive prediction report with probabilities and risk tiers.
    """
    # 1. Convert input dictionary to DataFrame
    df_single = pd.DataFrame([patient_data])
    
    # Ensure all required features exist
    for col in FEATURE_COLS:
        if col not in df_single.columns:
            raise ValueError(f"Missing required clinical feature: '{col}'")

    df_single = df_single[FEATURE_COLS].copy()

    # 2. Scale continuous features
    df_scaled = df_single.copy()
    df_scaled[CONTINUOUS_COLS] = scaler.transform(df_single[CONTINUOUS_COLS])
    X_single = df_scaled.values

    # 3. Model Inference
    # Random Forest
    rf_prob = float(rf_model.predict_proba(X_single)[0, 1])
    rf_pred = int(rf_prob >= 0.5)

    # Keras ANN
    ann_prob = float(ann_model.predict(X_single, verbose=0)[0, 0])
    ann_pred = int(ann_prob >= 0.5)

    # Ensemble Average Risk Score
    avg_prob = (rf_prob + ann_prob) / 2.0
    
    # Determine Risk Tier
    if avg_prob >= 0.65:
        risk_tier = "HIGH RISK"
        color_code = "RED"
        recommendation = "Immediate cardiology consultation and comprehensive Diagnostic Evaluation recommended."
    elif avg_prob >= 0.35:
        risk_tier = "MODERATE RISK"
        color_code = "YELLOW"
        recommendation = "Lifestlye modifications, regular monitoring, and follow-up clinical screening advised."
    else:
        risk_tier = "LOW RISK"
        color_code = "GREEN"
        recommendation = "Routine wellness monitoring and healthy cardiovascular habits maintained."

    report = {
        "patient_profile": patient_data,
        "random_forest": {
            "prediction": "High Risk (1)" if rf_pred == 1 else "Low Risk (0)",
            "probability": rf_prob
        },
        "keras_ann": {
            "prediction": "High Risk (1)" if ann_pred == 1 else "Low Risk (0)",
            "probability": ann_prob
        },
        "ensemble_risk": {
            "average_probability": avg_prob,
            "risk_tier": risk_tier,
            "color": color_code,
            "clinical_recommendation": recommendation
        }
    }

    return report


def print_prediction_report(report: Dict[str, Any], patient_name: str = "Patient Profile") -> None:
    """Prints a clean, formatted clinical inference report."""
    print("\n=======================================================")
    print(f"  HEART DISEASE RISK ASSESSMENT: {patient_name.upper()}")
    print("=======================================================")
    
    print("\n--- Clinical Metrics Provided ---")
    p = report["patient_profile"]
    print(f"  Age: {p['age']} yrs | Sex: {'Male (1)' if p['sex']==1 else 'Female (0)'}")
    print(f"  Chest Pain Type (cp): {p['cp']} | Resting BP: {p['trestbps']} mm Hg | Cholesterol: {p['chol']} mg/dl")
    print(f"  Fasting Blood Sugar > 120: {p['fbs']} | Resting ECG: {p['restecg']} | Max Heart Rate: {p['thalach']} bpm")
    print(f"  Exercise Angina: {p['exang']} | ST Depression (oldpeak): {p['oldpeak']} | ST Slope: {p['slope']}")
    print(f"  Major Vessels (ca): {p['ca']} | Thalium Test: {p['thal']}")

    print("\n--- Model Predictions ---")
    rf = report["random_forest"]
    print(f"  • Random Forest Classifier : {rf['prediction']} (Confidence: {rf['probability']:.1%})")

    ann = report["keras_ann"]
    print(f"  • Keras Neural Network     : {ann['prediction']} (Confidence: {ann['probability']:.1%})")

    ens = report["ensemble_risk"]
    print("\n--- Final Risk Tier Assessment ---")
    print(f"  >> Average Risk Probability: {ens['average_probability']:.1%}")
    print(f"  >> Clinical Risk Level     : [{ens['risk_tier']}]")
    print(f"  >> Actionable Advisory     : {ens['clinical_recommendation']}")
    print("=======================================================\n")
