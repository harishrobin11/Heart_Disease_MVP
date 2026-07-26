"""
FastAPI REST API Server for Serving Real-Time Heart Disease Risk Inference.

Loads serialized artifacts ('artifacts/scaler.pkl' and 'artifacts/random_forest_model.pkl')
and exposes a POST /predict endpoint for real-time model inference.
"""

import os
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Production-grade REST API for serving ML predictions on patient clinical profiles.",
    version="1.0.0"
)

# Global model and scaler variables
scaler = None
rf_model = None

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", 
    "ca", "thal"
]


class PatientPayload(BaseModel):
    """Pydantic schema for patient clinical feature input payload."""
    age: int = Field(..., ge=1, le=120, description="Patient age in years (e.g., 63)", example=63)
    sex: int = Field(..., ge=0, le=1, description="Gender (0 = Female, 1 = Male)", example=1)
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0: typical, 1: atypical, 2: non-anginal, 3: asymptomatic)", example=0)
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure in mm Hg", example=160)
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol in mg/dl", example=286)
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (0 = False, 1 = True)", example=1)
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0: Normal, 1: ST-T wave abnormality, 2: Hypertrophy)", example=1)
    thalach: int = Field(..., ge=50, le=250, description="Maximum heart rate achieved in bpm", example=108)
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina (0 = No, 1 = Yes)", example=1)
    oldpeak: float = Field(..., ge=0.0, le=10.0, description="ST depression induced by exercise relative to rest", example=2.6)
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0: Upsloping, 1: Flat, 2: Downsloping)", example=1)
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels (0-4) colored by fluoroscopy", example=2)
    thal: int = Field(..., ge=0, le=3, description="Thalium stress test (0: Normal, 1: Fixed defect, 2: Reversible defect, 3: Unknown)", example=2)


@app.on_event("startup")
def load_artifacts():
    """Loads serialized scaler and model artifacts into memory upon API startup."""
    global scaler, rf_model
    scaler_path = os.path.join("artifacts", "scaler.pkl")
    model_path = os.path.join("artifacts", "random_forest_model.pkl")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        # Auto-run main pipeline if artifacts are missing
        print("[API STARTUP] Artifacts missing. Running pipeline orchestrator main.py...")
        from main import main as run_main
        run_main()

    scaler = joblib.load(scaler_path)
    rf_model = joblib.load(model_path)
    print("[API STARTUP] Scaler and Random Forest model loaded successfully into API memory.")


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Health check endpoint confirming API status and model readiness."""
    return {
        "status": "healthy",
        "model_loaded": rf_model is not None,
        "scaler_loaded": scaler is not None,
        "message": "Heart Disease Risk Prediction API is operational."
    }


@app.post("/predict", status_code=status.HTTP_200_OK)
def predict(payload: PatientPayload):
    """
    POST Endpoint for predicting heart disease risk from JSON clinical input payload.

    Returns:
        JSON response with predicted risk level ('High Risk' or 'Low Risk'), 
        probability score, and clinical advisory.
    """
    if scaler is None or rf_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Models not initialized. Please ensure artifacts exist in 'artifacts/' folder."
        )

    try:
        # Convert payload to DataFrame
        input_dict = payload.dict()
        df_single = pd.DataFrame([input_dict])[FEATURE_COLS]

        # Apply fitted StandardScaler to continuous columns
        df_scaled = df_single.copy()
        df_scaled[CONTINUOUS_COLS] = scaler.transform(df_single[CONTINUOUS_COLS])
        X_input = df_scaled.values

        # Perform inference
        probability = float(rf_model.predict_proba(X_input)[0, 1])
        prediction_label = "High Risk" if probability >= 0.5 else "Low Risk"

        return {
            "status": "success",
            "prediction": prediction_label,
            "probability": round(probability, 4),
            "risk_score_pct": f"{probability * 100:.1f}%",
            "clinical_advisory": (
                "Immediate cardiology consultation and diagnostic evaluation recommended."
                if prediction_label == "High Risk"
                else "Routine wellness monitoring and healthy cardiovascular habits maintained."
            ),
            "patient_features": input_dict
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
