import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="AI Heart Predictor", page_icon="🏥")

st.title("🏥 AI Heart Disease Risk Predictor")
st.markdown("### Predict your heart disease risk using AI")

# Load model
model = joblib.load('models/heart_disease_model.pkl')

# Input form
age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex", ["Female", "Male"])
chol = st.number_input("Cholesterol (mg/dl)", 50, 1000, 200)
thalach = st.number_input("Max Heart Rate", 50, 300, 150)

if st.button("Predict Risk"):
    user_input = {
        'age': age, 'sex': 1 if sex == "Male" else 0,
        'cp': 2, 'trestbps': 120, 'chol': chol,
        'fbs': 0, 'restecg': 1, 'thalach': thalach,
        'exang': 0, 'oldpeak': 1.0, 'slope': 2,
        'ca': 0, 'thal': 2
    }
    
    prediction = model.predict(pd.DataFrame([user_input]))[0]
    probability = model.predict_proba(pd.DataFrame([user_input]))[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ HIGH RISK ({probability:.1%})")
    else:
        st.success(f"✅ LOW RISK ({probability:.1%})")
