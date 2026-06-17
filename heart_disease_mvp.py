"""
AI Disease Risk Predictor - MVP (Week 1)
Heart Disease Prediction with Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os

def load_heart_disease_data():
    
    df = pd.read_csv('heart.csv') 
    return pd.DataFrame(df)

def train_model(df):
    """Train Random Forest model"""
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']),
        ('cat', OneHotEncoder(), ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight='balanced'))
    ])
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    
    print(f"Accuracy: {accuracy:.2%}")
    print(f"AUC-ROC: {auc:.2%}")
    
    return model, accuracy, auc

if __name__ == "__main__":
    df = load_heart_disease_data()
    model, accuracy, auc = train_model(df)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/heart_disease_model.pkl')
    
    print("\n✅ MVP Complete! Model saved to models/heart_disease_model.pkl")
    print("Next: pip install streamlit && streamlit run app.py")
