"""
Model Training & Serialization Module for Heart Disease Risk MLOps Architecture.

Contains functions to train:
1. Random Forest Classifier via GridSearchCV (saved to 'artifacts/random_forest_model.pkl').
2. Artificial Neural Network via Keras (saved to 'artifacts/ann_model.h5').
"""

import os
from typing import Tuple, Dict, Any
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 42,
    cv: int = 5,
    artifact_dir: str = "artifacts"
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Performs GridSearchCV hyperparameter tuning on Random Forest Classifier
    and serializes the best fitted model to '{artifact_dir}/random_forest_model.pkl'.

    Parameters:
        X_train (np.ndarray): Preprocessed training features.
        y_train (np.ndarray): Training labels.
        random_state (int): Seed for reproducibility.
        cv (int): Cross-validation folds.
        artifact_dir (str): Folder path for saving serialized model.

    Returns:
        best_rf (RandomForestClassifier): Tuned Random Forest estimator.
        best_params (Dict[str, Any]): Dictionary of best hyperparameters.
    """
    print("\n==========================================")
    print("     TRAINING MODEL A: RANDOM FOREST      ")
    print("==========================================")

    os.makedirs(artifact_dir, exist_ok=True)

    rf = RandomForestClassifier(random_state=random_state)
    
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 8, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    print("[TRAIN] Executing 5-Fold GridSearchCV tuning for Random Forest...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,  # n_jobs=1 avoids macOS fork deadlock with TF
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"[TRAIN] Random Forest GridSearch Complete! Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"[TRAIN] Best Parameters: {best_params}")

    # Serialize best Random Forest model using joblib
    model_path = os.path.join(artifact_dir, "random_forest_model.pkl")
    joblib.dump(best_rf, model_path)
    print(f"[TRAIN] Serialized best Random Forest model to '{model_path}'")

    return best_rf, best_params


def train_ann_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    random_state: int = 42,
    artifact_dir: str = "artifacts"
) -> Tuple[Sequential, tf.keras.callbacks.History]:
    """
    Builds, compiles, and trains an Artificial Neural Network using Keras:
    - Dense Input layer (32 units, ReLU, Dropout 0.2)
    - Hidden Dense layer (16 units, ReLU, Batch Normalization)
    - Output Dense layer (1 unit, Sigmoid for binary risk classification)
    - Adam optimizer & binary_crossentropy loss
    - EarlyStopping callback
    Saves trained model as '{artifact_dir}/ann_model.h5'.

    Parameters:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training labels.
        input_dim (int): Input feature dimension.
        epochs (int): Max epochs.
        batch_size (int): Batch size.
        validation_split (float): Validation split ratio.
        random_state (int): Seed for TF.
        artifact_dir (str): Target directory for saving model.

    Returns:
        model (Sequential): Trained Keras ANN model.
        history (History): Training metrics history.
    """
    print("\n==========================================")
    print("      TRAINING MODEL B: KERAS ANN         ")
    print("==========================================")

    os.makedirs(artifact_dir, exist_ok=True)

    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    model = Sequential([
        Dense(32, activation="relu", input_shape=(input_dim,), name="input_dense"),
        Dropout(0.2, name="dropout_layer"),
        Dense(16, activation="relu", name="hidden_dense"),
        BatchNormalization(name="batch_norm"),
        Dense(1, activation="sigmoid", name="output_sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", "auc"]
    )

    print("\n--- ANN Architecture Summary ---")
    model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    print("[TRAIN] Fitting Artificial Neural Network with EarlyStopping...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0
    )

    # Save model artifact in HDF5 format
    ann_path = os.path.join(artifact_dir, "ann_model.h5")
    model.save(ann_path)
    print(f"[TRAIN] Serialized Keras ANN model to '{ann_path}'")

    return model, history
