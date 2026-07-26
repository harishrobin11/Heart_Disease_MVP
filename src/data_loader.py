"""
Data Loader Module for MLOps Heart Disease Risk Pipeline.

Handles checking local raw CSV storage ('data/raw/heart_disease.csv') or synthetically 
generating a domain-realistic 1,000-row DataFrame with standard clinical features.
"""

import os
import pandas as pd
import numpy as np


def generate_synthetic_heart_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic DataFrame containing 13 clinical features
    and binary target variable for heart disease prediction.
    """
    np.random.seed(random_state)

    age = np.random.randint(29, 78, size=n_samples)
    sex = np.random.choice([0, 1], size=n_samples, p=[0.32, 0.68])
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.45, 0.17, 0.28, 0.10])
    trestbps = np.random.normal(loc=131, scale=17.5, size=n_samples).clip(94, 200).astype(int)
    chol = np.random.normal(loc=246, scale=50, size=n_samples).clip(126, 564).astype(int)
    fbs = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.48, 0.48, 0.04])
    
    base_thalach = 220 - age + np.random.normal(0, 15, size=n_samples)
    thalach = base_thalach.clip(71, 202).astype(int)

    exang_prob = np.where(cp == 0, 0.45, 0.20)
    exang = np.random.binomial(1, exang_prob)
    oldpeak = np.random.exponential(scale=1.0, size=n_samples).clip(0.0, 6.2).round(1)
    slope = np.random.choice([0, 1, 2], size=n_samples, p=[0.07, 0.46, 0.47])
    ca = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.57, 0.21, 0.12, 0.07, 0.03])
    thal = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.02, 0.54, 0.38, 0.06])

    # Realistic target distribution via logistic log-odds calculation
    logit = (
        - 2.5
        + 0.03 * (age - 50)
        + 0.60 * (cp > 0)
        + 0.015 * (trestbps - 120)
        + 0.005 * (chol - 200)
        - 0.03 * (thalach - 150)
        + 0.80 * exang
        + 0.50 * oldpeak
        + 0.60 * ca
        + 0.40 * (thal == 2)
        + np.random.normal(0, 0.8, size=n_samples)
    )
    prob = 1 / (1 + np.exp(-logit))
    target = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target
    })

    return df


def load_or_generate_data(filepath: str = "data/raw/heart_disease.csv", n_samples: int = 1000) -> pd.DataFrame:
    """
    Checks if raw dataset CSV exists at specified filepath.
    If missing, generates a realistic 1,000-row synthetic clinical DataFrame 
    and saves it to 'data/raw/heart_disease.csv'.

    Parameters:
        filepath (str): Target CSV filepath (default: 'data/raw/heart_disease.csv').
        n_samples (int): Number of synthetic rows to generate if CSV missing.

    Returns:
        pd.DataFrame: Cleaned clinical dataset DataFrame.
    """
    if os.path.exists(filepath):
        print(f"[DATA LOADER] Loading raw dataset from '{filepath}'...")
        df = pd.read_csv(filepath)
        print(f"[DATA LOADER] Dataset loaded successfully ({df.shape[0]} rows, {df.shape[1]} columns).")
        return df

    print(f"[DATA LOADER] Target file '{filepath}' not found. Generating synthetic dataset...")
    df = generate_synthetic_heart_data(n_samples=n_samples, random_state=42)

    # Ensure parent directory exists (data/raw/)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"[DATA LOADER] Synthetic dataset generated and saved to '{filepath}'")

    return df
