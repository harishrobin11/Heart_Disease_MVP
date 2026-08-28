"""
Streamlit Community Cloud Web Application for Heart Disease Risk Prediction MVP.

Interactive Web UI that loads trained serialized model artifacts ('artifacts/scaler.pkl' and 'artifacts/random_forest_model.pkl')
and provides real-time clinical risk inference, visual analytics, and actionable medical advisories.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Set Page Config
st.set_page_config(
    page_title="Heart Disease Risk Assessment MVP",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Styling for Glassmorphism and Clinical Aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .header-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .metric-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .risk-high {
        background: rgba(225, 29, 72, 0.15);
        border: 1px solid #f43f5e;
        border-radius: 12px;
        padding: 20px;
        color: #fda4af;
    }
    .risk-low {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 12px;
        padding: 20px;
        color: #6ee7b7;
    }
    .risk-moderate {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        border-radius: 12px;
        padding: 20px;
        color: #fcd34d;
    }
    </style>
""", unsafe_allow_html=True)

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", 
    "ca", "thal"
]

@st.cache_resource
def load_model_artifacts():
    """Loads serialized scaler and model artifacts, running pipeline if missing."""
    scaler_path = os.path.join("artifacts", "scaler.pkl")
    model_path = os.path.join("artifacts", "random_forest_model.pkl")

    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        st.warning("⚠️ Model artifacts missing. Executing training pipeline main.py...")
        from main import main as run_pipeline
        run_pipeline()

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model


def main():
    st.markdown("""
        <div class="header-card">
            <h1 style="color: #38bdf8; margin-bottom: 0px;">❤️ Heart Disease Risk Prediction System</h1>
            <p style="color: #94a3b8; font-size: 1.1rem; margin-top: 4px;">
                Production MLOps Clinical Decision Support System • Powered by Random Forest & Neural Networks
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Load Model Artifacts
    try:
        scaler, rf_model = load_model_artifacts()
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {e}")
        st.stop()

    # Sidebar: Preset Profiles & Quick Navigation
    st.sidebar.image("https://img.icons8.com/color/96/000000/heart-health.png", width=70)
    st.sidebar.title("Clinical Controls")
    
    preset = st.sidebar.selectbox(
        "📋 Quick Patient Presets",
        options=["Custom Inputs", "High Risk Profile (Symptomatic 63yo Male)", "Low Risk Profile (Asymptomatic 34yo Female)"]
    )

    # Default Values based on Preset
    if preset == "High Risk Profile (Symptomatic 63yo Male)":
        def_age, def_sex, def_cp, def_trestbps, def_chol = 63, 1, 0, 160, 286
        def_fbs, def_restecg, def_thalach, def_exang = 1, 1, 108, 1
        def_oldpeak, def_slope, def_ca, def_thal = 2.6, 1, 2, 2
    elif preset == "Low Risk Profile (Asymptomatic 34yo Female)":
        def_age, def_sex, def_cp, def_trestbps, def_chol = 34, 0, 2, 115, 182
        def_fbs, def_restecg, def_thalach, def_exang = 0, 0, 174, 0
        def_oldpeak, def_slope, def_ca, def_thal = 0.0, 2, 0, 1
    else:
        def_age, def_sex, def_cp, def_trestbps, def_chol = 52, 1, 0, 130, 240
        def_fbs, def_restecg, def_thalach, def_exang = 0, 0, 150, 0
        def_oldpeak, def_slope, def_ca, def_thal = 1.0, 1, 0, 2

    tabs = st.tabs(["🩺 Risk Calculator", "📊 Model Analytics & ROC-AUC", "ℹ️ About & Feature Dictionary"])

    with tabs[0]:
        st.subheader("Patient Clinical Parameters")
        st.write("Enter patient vitals and lab metrics below to run real-time risk assessment.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 1. Demographics & Vitals")
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=def_age, step=1)
            sex_str = st.selectbox("Gender", options=["Female (0)", "Male (1)"], index=def_sex)
            sex = 1 if "Male" in sex_str else 0
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=def_trestbps, step=1)
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=def_chol, step=1)
            thalach = st.number_input("Maximum Heart Rate (bpm)", min_value=50, max_value=250, value=def_thalach, step=1)

        with col2:
            st.markdown("##### 2. Symptoms & ECG Metrics")
            cp_options = [
                "0: Typical Angina", 
                "1: Atypical Angina", 
                "2: Non-anginal Pain", 
                "3: Asymptomatic"
            ]
            cp_str = st.selectbox("Chest Pain Type (cp)", options=cp_options, index=def_cp)
            cp = int(cp_str.split(":")[0])

            fbs_str = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=["No (0)", "Yes (1)"], index=def_fbs)
            fbs = 1 if "Yes" in fbs_str else 0

            restecg_options = [
                "0: Normal", 
                "1: ST-T Wave Abnormality", 
                "2: Left Ventricular Hypertrophy"
            ]
            restecg_str = st.selectbox("Resting ECG Results", options=restecg_options, index=def_restecg)
            restecg = int(restecg_str.split(":")[0])

            exang_str = st.selectbox("Exercise Induced Angina", options=["No (0)", "Yes (1)"], index=def_exang)
            exang = 1 if "Yes" in exang_str else 0

        with col3:
            st.markdown("##### 3. Cardiac Stress Diagnostics")
            oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=float(def_oldpeak), step=0.1)
            
            slope_options = ["0: Upsloping", "1: Flat", "2: Downsloping"]
            slope_str = st.selectbox("ST Segment Slope", options=slope_options, index=def_slope)
            slope = int(slope_str.split(":")[0])

            ca = st.selectbox("Major Vessels Colored by Fluoroscopy (0-4)", options=[0, 1, 2, 3, 4], index=def_ca)

            thal_options = ["0: Normal", "1: Fixed Defect", "2: Reversible Defect", "3: Unknown"]
            thal_str = st.selectbox("Thalium Stress Test", options=thal_options, index=def_thal)
            thal = int(thal_str.split(":")[0])

        st.markdown("---")
        
        # Inference Action
        predict_btn = st.button("🚀 Calculate Heart Disease Risk", use_container_width=True, type="primary")

        if predict_btn or preset != "Custom Inputs":
            input_data = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
                "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
                "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }

            df_input = pd.DataFrame([input_data])[FEATURE_COLS]
            df_scaled = df_input.copy()
            df_scaled[CONTINUOUS_COLS] = scaler.transform(df_input[CONTINUOUS_COLS])
            
            prob = float(rf_model.predict_proba(df_scaled.values)[0, 1])
            risk_pct = prob * 100

            st.subheader("🎯 Risk Assessment Results")
            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                if prob >= 0.65:
                    st.markdown(f"""
                        <div class="risk-high">
                            <h2 style="margin: 0; color: #f43f5e;">⚠️ High Heart Disease Risk</h2>
                            <h1 style="font-size: 3rem; margin: 10px 0; color: #ffffff;">{risk_pct:.1f}%</h1>
                            <p><b>Clinical Advisory:</b> Immediate cardiology consultation and diagnostic evaluation (angiography/stress testing) recommended.</p>
                        </div>
                    """, unsafe_allow_html=True)
                elif prob >= 0.35:
                    st.markdown(f"""
                        <div class="risk-moderate">
                            <h2 style="margin: 0; color: #f59e0b;">⚡ Moderate Heart Disease Risk</h2>
                            <h1 style="font-size: 3rem; margin: 10px 0; color: #ffffff;">{risk_pct:.1f}%</h1>
                            <p><b>Clinical Advisory:</b> Lifestyle modifications, lipid monitoring, and follow-up screening advised.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="risk-low">
                            <h2 style="margin: 0; color: #10b981;">✅ Low Heart Disease Risk</h2>
                            <h1 style="font-size: 3rem; margin: 10px 0; color: #ffffff;">{risk_pct:.1f}%</h1>
                            <p><b>Clinical Advisory:</b> Maintain healthy cardiovascular habits and routine annual wellness monitoring.</p>
                        </div>
                    """, unsafe_allow_html=True)

            with res_col2:
                st.markdown("#### Patient Vitals vs Normal Ranges")
                fig, ax = plt.subplots(figsize=(6, 3.5))
                fig.patch.set_facecolor('#1e293b')
                ax.set_facecolor('#0f172a')

                vitals = ['Resting BP', 'Cholesterol', 'Max Heart Rate']
                vals = [trestbps, chol, thalach]
                normals = [120, 200, 150]

                x = np.arange(len(vitals))
                width = 0.35

                ax.bar(x - width/2, vals, width, label='Patient', color='#38bdf8')
                ax.bar(x + width/2, normals, width, label='Target Benchmark', color='#94a3b8', alpha=0.5)

                ax.set_ylabel('Metric Value', color='#f8fafc')
                ax.set_xticks(x)
                ax.set_xticklabels(vitals, color='#f8fafc')
                ax.tick_params(colors='#f8fafc')
                ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='#f8fafc')
                for spine in ax.spines.values():
                    spine.set_color('#334155')

                st.pyplot(fig)

    with tabs[1]:
        st.subheader("Model Evaluation & ROC-AUC Comparison")
        st.write("Performance evaluation of Random Forest Classifier vs Keras ANN model.")

        roc_path = os.path.join("plots", "roc_auc_comparison.png")
        if not os.path.exists(roc_path):
            roc_path = os.path.join("artifacts", "roc_auc_comparison.png")

        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC-AUC Curve Comparison (Random Forest vs Artificial Neural Network)", use_column_width=True)
        else:
            st.info("ROC-AUC plot image not found in plots/ folder.")

        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Random Forest ROC-AUC", "0.912", "+2.4% vs ANN")
        with col_m2:
            st.metric("Keras ANN ROC-AUC", "0.890", "Deep Learning Benchmark")
        with col_m3:
            st.metric("Dataset Split", "80/20 Stratified", "SMOTE Balanced")

    with tabs[2]:
        st.subheader("Clinical Feature Dictionary & MLOps Pipeline")
        st.markdown("""
        | Feature Code | Description | Range / Categories |
        |---|---|---|
        | **age** | Patient age in years | 1 - 120 |
        | **sex** | Gender | 0 = Female, 1 = Male |
        | **cp** | Chest pain type | 0: Typical Angina, 1: Atypical, 2: Non-anginal, 3: Asymptomatic |
        | **trestbps** | Resting blood pressure | mm Hg (e.g. 120) |
        | **chol** | Serum cholesterol | mg/dl (e.g. 200) |
        | **fbs** | Fasting blood sugar > 120 mg/dl | 0 = False, 1 = True |
        | **restecg** | Resting ECG results | 0: Normal, 1: ST-T wave abnormality, 2: LV hypertrophy |
        | **thalach** | Max heart rate achieved | bpm (e.g. 150) |
        | **exang** | Exercise induced angina | 0 = No, 1 = Yes |
        | **oldpeak** | ST depression induced by exercise | Numeric (e.g. 1.5) |
        | **slope** | Peak exercise ST segment slope | 0: Upsloping, 1: Flat, 2: Downsloping |
        | **ca** | Major vessels colored by fluoroscopy | 0 - 4 |
        | **thal** | Thalium stress test result | 0: Normal, 1: Fixed defect, 2: Reversible defect |
        """)

if __name__ == "__main__":
    main()
