"""
Interactive & Batch Patient Risk Prediction Script.

Demonstrates real-time heart disease risk scoring on sample clinical patient profiles
using the trained Random Forest and Keras ANN models.
"""

import sys
import argparse
from src.data_loader import load_or_generate_data
from src.preprocessing import prepare_data
from src.models import train_random_forest, build_and_train_ann
from src.predict import predict_patient_risk, print_prediction_report


def run_prediction_demo():
    print("\n=======================================================")
    print("        HEART DISEASE RISK PREDICTION INFERENCE         ")
    print("=======================================================")

    # 1. Load Data & Train Pipeline Models
    print("\n[STEP 1/2] Loading clinical dataset & training models...")
    df = load_or_generate_data(random_state=42)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df, use_smote=True)

    rf_model, _ = train_random_forest(X_train, y_train)
    ann_model, _ = build_and_train_ann(X_train, y_train, input_dim=X_train.shape[1], epochs=50)

    print("\n[STEP 2/2] Running Patient Risk Inference...")

    # Sample Patient 1: High Risk Case Profile
    patient_high_risk = {
        "age": 63,
        "sex": 1,          # Male
        "cp": 0,           # Typical angina
        "trestbps": 160,   # Elevated blood pressure (160 mm Hg)
        "chol": 286,       # High cholesterol (286 mg/dl)
        "fbs": 1,          # Fasting blood sugar > 120 mg/dl
        "restecg": 1,      # ST-T wave abnormality
        "thalach": 108,    # Low max heart rate (108 bpm)
        "exang": 1,        # Exercise induced angina (Yes)
        "oldpeak": 2.6,    # ST depression (2.6)
        "slope": 1,        # Flat ST slope
        "ca": 2,           # 2 major vessels colored
        "thal": 2          # Reversible defect
    }

    report_high = predict_patient_risk(patient_high_risk, rf_model, ann_model, scaler)
    print_prediction_report(report_high, patient_name="Patient #1 (Symptomatic 63yo Male)")

    # Sample Patient 2: Low Risk Case Profile
    patient_low_risk = {
        "age": 34,
        "sex": 0,          # Female
        "cp": 2,           # Non-anginal pain
        "trestbps": 115,   # Normal blood pressure (115 mm Hg)
        "chol": 182,       # Healthy cholesterol (182 mg/dl)
        "fbs": 0,          # Fasting blood sugar normal
        "restecg": 0,      # Normal ECG
        "thalach": 174,    # High max heart rate (174 bpm)
        "exang": 0,        # No exercise angina
        "oldpeak": 0.0,    # Normal ST segment
        "slope": 2,        # Upsloping ST segment
        "ca": 0,           # 0 major vessels
        "thal": 1          # Normal thalium test
    }

    report_low = predict_patient_risk(patient_low_risk, rf_model, ann_model, scaler)
    print_prediction_report(report_low, patient_name="Patient #2 (Asymptomatic 34yo Female)")


if __name__ == "__main__":
    run_prediction_demo()
