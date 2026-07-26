"""
Preprocessing Pipeline Module for Heart Disease MLOps Architecture.

Performs stratified train/test split, fits StandardScaler on continuous features,
serializes the scaler to 'artifacts/scaler.pkl', and balances training set using SMOTE.
"""

import os
from typing import Tuple, List
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

CONTINUOUS_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
    artifact_dir: str = "artifacts"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Executes the preprocessing workflow:
    1. Stratified 80/20 train/test split.
    2. Continuous feature scaling using StandardScaler.
    3. Serializing fitted StandardScaler to '{artifact_dir}/scaler.pkl'.
    4. SMOTE oversampling on training data for class balance.

    Parameters:
        df (pd.DataFrame): Input clinical DataFrame.
        target_col (str): Target column name.
        test_size (float): Proportion of test split (default: 0.2).
        random_state (int): Seed for reproducibility.
        artifact_dir (str): Folder path to save serialized scaler artifact.

    Returns:
        X_train_res (np.ndarray): SMOTE-balanced & scaled training features.
        X_test_scaled (np.ndarray): Scaled test features.
        y_train_res (np.ndarray): Training target labels.
        y_test (np.ndarray): Unaltered test target labels.
        feature_names (List[str]): List of input feature names.
    """
    print("\n==========================================")
    print("      RUNNING PREPROCESSING PIPELINE       ")
    print("==========================================")

    os.makedirs(artifact_dir, exist_ok=True)

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    feature_names = list(X.columns)

    # 1. Stratified Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[PREPROCESSING] Stratified Train-Test Split (80/20):")
    print(f"  - Train Set: {X_train.shape[0]} samples")
    print(f"  - Test Set:  {X_test.shape[0]} samples")

    # 2. Fit StandardScaler on continuous features
    scaler = StandardScaler()
    
    X_train_scaled_df = X_train.copy()
    X_test_scaled_df = X_test.copy()

    X_train_scaled_df[CONTINUOUS_COLS] = scaler.fit_transform(X_train[CONTINUOUS_COLS])
    X_test_scaled_df[CONTINUOUS_COLS] = scaler.transform(X_test[CONTINUOUS_COLS])

    X_train_scaled = X_train_scaled_df.values
    X_test_scaled = X_test_scaled_df.values

    # 3. Save fitted StandardScaler artifact using joblib
    scaler_path = os.path.join(artifact_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[PREPROCESSING] Saved fitted StandardScaler to '{scaler_path}'")

    # 4. Handle Class Imbalance using SMOTE
    y_train_arr = y_train.values
    print("[PREPROCESSING] Applying SMOTE oversampling to training set...")
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train_arr)
    print(f"[PREPROCESSING] Training data balanced: {X_train_res.shape[0]} samples (Class 0: {(y_train_res==0).sum()}, Class 1: {(y_train_res==1).sum()})")

    return X_train_res, X_test_scaled, y_train_res, y_test.values, feature_names
