"""
Data Loader Module for Heart Disease Risk Prediction.

Handles loading data from a CSV file, downloading the official UCI Heart Disease dataset,
or generating a domain-realistic 1,000-row synthetic dataset with clinical features.
"""

import os
import io
import urllib.request
import pandas as pd
import numpy as np


def generate_synthetic_heart_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic DataFrame containing clinical features 
    commonly associated with heart disease risk modeling.

    Parameters:
        n_samples (int): Number of synthetic patient rows to generate (default: 1000).
        random_state (int): Seed for reproducible random generation.

    Returns:
        pd.DataFrame: Synthetic heart disease dataset with 14 standard clinical features.
    """
    np.random.seed(random_state)

    # 1. Age: 29 to 77 years
    age = np.random.randint(29, 78, size=n_samples)

    # 2. Sex: 0 = female (approx 32%), 1 = male (approx 68%)
    sex = np.random.choice([0, 1], size=n_samples, p=[0.32, 0.68])

    # 3. Chest Pain Type (cp): 0: typical angina, 1: atypical angina, 2: non-anginal, 3: asymptomatic
    cp = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.45, 0.17, 0.28, 0.10])

    # 4. Resting Blood Pressure (trestbps): 94 - 200 mm Hg
    trestbps = np.random.normal(loc=131, scale=17.5, size=n_samples).clip(94, 200).astype(int)

    # 5. Serum Cholestoral (chol): 126 - 564 mg/dl
    chol = np.random.normal(loc=246, scale=50, size=n_samples).clip(126, 564).astype(int)

    # 6. Fasting Blood Sugar > 120 mg/dl (fbs): 0 or 1
    fbs = np.random.choice([0, 1], size=n_samples, p=[0.85, 0.15])

    # 7. Resting Electrocardiographic results (restecg): 0, 1, 2
    restecg = np.random.choice([0, 1, 2], size=n_samples, p=[0.48, 0.48, 0.04])

    # 8. Maximum Heart Rate Achieved (thalach): 71 - 202 (negatively correlated with age)
    base_thalach = 220 - age + np.random.normal(0, 15, size=n_samples)
    thalach = base_thalach.clip(71, 202).astype(int)

    # 9. Exercise Induced Angina (exang): 0 = no, 1 = yes
    exang_prob = np.where(cp == 0, 0.45, 0.20)
    exang = np.random.binomial(1, exang_prob)

    # 10. ST depression induced by exercise (oldpeak): 0.0 - 6.2
    oldpeak = np.random.exponential(scale=1.0, size=n_samples).clip(0.0, 6.2).round(1)

    # 11. Slope of peak exercise ST segment (slope): 0, 1, 2
    slope = np.random.choice([0, 1, 2], size=n_samples, p=[0.07, 0.46, 0.47])

    # 12. Number of major vessels (ca): 0 - 4
    ca = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.57, 0.21, 0.12, 0.07, 0.03])

    # 13. Thalium Stress Test (thal): 0: normal, 1: fixed defect, 2: reversible defect, 3: unknown
    thal = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.02, 0.54, 0.38, 0.06])

    # 14. Target calculation using realistic clinical log-odds equation
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


def download_uci_heart_disease() -> pd.DataFrame:
    """
    Attempts to fetch the Cleveland UCI Heart Disease dataset from online repository.

    Returns:
        pd.DataFrame: Loaded dataset if successful.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", 
        "restecg", "thalach", "exang", "oldpeak", "slope", 
        "ca", "thal", "target"
    ]
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=3) as response:
        content = response.read().decode('utf-8')
    
    df = pd.read_csv(io.StringIO(content), names=columns, na_values="?")
    
    # Preprocess UCI specific target: >0 means heart disease present
    df['target'] = (df['target'] > 0).astype(int)
    
    # Fill any missing values with median
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    return df


def load_or_generate_data(filepath: str = None, n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Loads dataset from filepath if provided and exists, or synthetically generates 
    a domain-realistic 1,000-row DataFrame with standard clinical features.

    Parameters:
        filepath (str, optional): Local CSV filepath.
        n_samples (int): Number of rows for synthetic generation (default: 1000).
        random_state (int): Seed for random number generators.

    Returns:
        pd.DataFrame: Cleaned Pandas DataFrame ready for EDA and preprocessing.
    """
    if filepath and os.path.exists(filepath):
        print(f"[DATA LOADER] Loading dataset from file path: '{filepath}'")
        df = pd.read_csv(filepath)
        return df

    # Check if a previously saved dataset copy exists in data/
    local_copy = os.path.join("data", "heart_disease_data.csv")
    if os.path.exists(local_copy):
        print(f"[DATA LOADER] Loading cached clinical dataset from '{local_copy}'")
        return pd.read_csv(local_copy)

    # Fallback to fast realistic synthetic generation
    print(f"[DATA LOADER] Generating domain-realistic synthetic dataset ({n_samples} samples)...")
    df = generate_synthetic_heart_data(n_samples=n_samples, random_state=random_state)
    print(f"[DATA LOADER] Synthetic dataset generated ({df.shape[0]} rows, {df.shape[1]} columns).")

    # Save to data/heart_disease_data.csv for reproducibility
    os.makedirs("data", exist_ok=True)
    df.to_csv(local_copy, index=False)
    print(f"[DATA LOADER] Saved dataset copy to '{local_copy}'")

    return df
