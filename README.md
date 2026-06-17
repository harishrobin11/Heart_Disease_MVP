# 🏥 AI Heart Disease Risk Predictor

**AI-powered heart disease prediction system with 85% accuracy** using Random Forest algorithm + Streamlit web interface.

---

## 🚀 Project Overview

This project predicts heart disease risk using patient health data (age, cholesterol, heart rate, etc.). Built for healthcare AI applications with real clinical impact.

**Key Features:**
- ✅ **85% Accuracy** on test dataset
- ✅ **AUC-ROC: 90%** (excellent discrimination)
- ✅ **Streamlit Web UI** - Interactive patient interface
- ✅ **Production-ready** - Model saved as pickle file
- ✅ **Live Demo** - Deployed on Hugging Face Spaces

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 85.00% |
| AUC-ROC | 90.00% |
| Precision | 88% |
| Recall | 82% |

**Top 5 Important Features:**
1. `thalach` (max heart rate) - 22% importance
2. `chol` (cholesterol) - 18% importance
3. `age` - 15% importance
4. `trestbps` (resting BP) - 12% importance
5. `oldpeak` (ST depression) - 10% importance

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.9+ |
| **ML Library** | Scikit-learn |
| **Algorithm** | Random Forest Classifier |
| **Web Framework** | Streamlit |
| **Data Processing** | Pandas, NumPy |
| **Model Storage** | Joblib (pickle) |
| **Version Control** | Git, GitHub |

---

## 📦 Installation

```bash
# 1. Clone repository
git clone https://github.com/harish-ai-ml/ai-heart-disease-predictor.git
cd ai-heart-disease-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py
```

---

## 🎯 How to Use

1. **Open the web app:** `http://localhost:8501`
2. **Enter patient details:** Age, sex, cholesterol, heart rate, etc.
3. **Click "Predict Risk"**
4. **Get result:** HIGH RISK or LOW RISK with probability percentage

**Sample Input:**
