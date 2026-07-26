"""
Preprocessing Pipeline Module for Heart Disease Risk Prediction.

Handles train/test splitting with stratification, feature scaling using StandardScaler,
and class imbalance balancing via SMOTE (Synthetic Minority Over-sampling Technique).
"""

from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
    use_smote: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, List[str]]:
    """
    Preprocesses the dataset by performing a stratified train/test split, 
    scaling continuous features with StandardScaler, and optionally balancing
    the training set using SMOTE.

    Parameters:
        df (pd.DataFrame): Raw clinical DataFrame.
        target_col (str): Name of the target label column.
        test_size (float): Proportion of test split (default: 0.2).
        random_state (int): Random seed for reproducibility.
        use_smote (bool): Whether to apply SMOTE oversampling on training data.

    Returns:
        X_train_final (np.ndarray): Processed and balanced training feature matrix.
        X_test_scaled (np.ndarray): Scaled test feature matrix.
        y_train_final (np.ndarray): Training target labels (balanced if SMOTE applied).
        y_test (np.ndarray): Unaltered test target labels.
        scaler (StandardScaler): Fitted StandardScaler instance.
        feature_names (List[str]): List of feature column names.
    """
    print("\n==========================================")
    print("         PREPROCESSING PIPELINE           ")
    print("==========================================")

    # Separate features and target
    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()
    feature_names = list(X.columns)

    # Continuous features to scale
    continuous_cols = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    # Ensure specified columns exist in dataframe
    continuous_cols = [c for c in continuous_cols if c in X.columns]

    print(f"[PREPROCESSING] Total Features ({len(feature_names)}): {feature_names}")
    print(f"[PREPROCESSING] Continuous Features for StandardScaler: {continuous_cols}")

    # 1. Stratified Train / Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"[PREPROCESSING] Stratified Train-Test Split (ratio {1-test_size:.0%}/{test_size:.0%}):")
    print(f"  - Train Set: {X_train.shape[0]} samples (Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()})")
    print(f"  - Test Set:  {X_test.shape[0]} samples (Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()})")

    # 2. StandardScaler Transformation on Continuous Features
    scaler = StandardScaler()
    
    # Create copies for scaling
    X_train_scaled_df = X_train.copy()
    X_test_scaled_df = X_test.copy()

    X_train_scaled_df[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled_df[continuous_cols] = scaler.transform(X_test[continuous_cols])

    X_train_scaled = X_train_scaled_df.values
    X_test_scaled = X_test_scaled_df.values

    # 3. Handle Potential Class Imbalance using SMOTE on Training Set
    y_train_arr = y_train.values
    if use_smote:
        class_0_count = (y_train_arr == 0).sum()
        class_1_count = (y_train_arr == 1).sum()
        imbalance_ratio = abs(class_0_count - class_1_count) / len(y_train_arr)

        if imbalance_ratio > 0.05:  # Apply SMOTE if imbalance > 5%
            print(f"[PREPROCESSING] Class imbalance detected. Applying SMOTE...")
            smote = SMOTE(random_state=random_state)
            X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train_arr)
            print(f"[PREPROCESSING] After SMOTE Balancing: {X_train_final.shape[0]} training samples")
            print(f"  - Balanced Class 0: {(y_train_final == 0).sum()}, Balanced Class 1: {(y_train_final == 1).sum()}")
        else:
            print("[PREPROCESSING] Classes are sufficiently balanced. Skipping SMOTE oversampling.")
            X_train_final, y_train_final = X_train_scaled, y_train_arr
    else:
        X_train_final, y_train_final = X_train_scaled, y_train_arr

    return X_train_final, X_test_scaled, y_train_final, y_test.values, scaler, feature_names
