"""
Model Building & Hyperparameter Tuning Module for Heart Disease Risk Prediction.

Includes:
- Model A: Random Forest Classifier hyperparameter-tuned with GridSearchCV.
- Model B: Artificial Neural Network (ANN) built with Keras (Dense ReLU, Dropout, Batch Normalization, EarlyStopping).
"""

from typing import Tuple, Dict, Any
import numpy as np
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
    cv: int = 5
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Trains a Random Forest Classifier and performs GridSearchCV to optimize hyperparameters.

    Parameters:
        X_train (np.ndarray): Preprocessed training features.
        y_train (np.ndarray): Training target labels.
        random_state (int): Seed for random forest reproducibility.
        cv (int): Number of cross-validation folds.

    Returns:
        best_rf (RandomForestClassifier): Tuned Random Forest model.
        best_params (Dict[str, Any]): Dictionary of best hyperparameters discovered.
    """
    print("\n==========================================")
    print("      MODEL A: RANDOM FOREST TUNING       ")
    print("==========================================")

    # Base model
    rf = RandomForestClassifier(random_state=random_state)

    # Streamlined Hyperparameter grid for fast & thorough GridSearchCV
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [4, 8, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    print(f"[RANDOM FOREST] Initiating 5-Fold GridSearchCV across grid parameters...")
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,  # n_jobs=1 avoids macOS fork deadlock with TensorFlow C++ runtime
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("[RANDOM FOREST] GridSearch Complete!")
    print(f">> Best ROC-AUC Cross-Validation Score: {grid_search.best_score_:.4f}")
    print(">> Best Hyperparameters Selected:")
    for param, val in best_params.items():
        print(f"   - {param}: {val}")

    return best_rf, best_params


def build_and_train_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int,
    epochs: int = 100,
    batch_size: int = 32,
    validation_split: float = 0.2,
    random_state: int = 42
) -> Tuple[Sequential, tf.keras.callbacks.History]:
    """
    Builds and trains an Artificial Neural Network (ANN) using Keras with:
    - Input Dense layer (32 units, ReLU activation, Dropout 0.2)
    - Hidden Dense layer (16 units, ReLU activation, BatchNormalization)
    - Output Dense layer (1 unit, Sigmoid activation for binary classification)
    - Adam optimizer and binary_crossentropy loss
    - EarlyStopping monitor restoring best weights

    Parameters:
        X_train (np.ndarray): Scaled training features.
        y_train (np.ndarray): Training target labels.
        input_dim (int): Number of input features.
        epochs (int): Max training epochs.
        batch_size (int): Mini-batch size.
        validation_split (float): Validation split ratio during training.
        random_state (int): Seed for TF reproducibility.

    Returns:
        model (Sequential): Trained Keras ANN model.
        history (History): Training loss and metrics history callback.
    """
    print("\n==========================================")
    print("      MODEL B: KERAS ANN ARCHITECTURE     ")
    print("==========================================")

    # Set seeds for Keras reproducibility
    tf.random.set_seed(random_state)
    np.random.seed(random_state)

    # Construct Sequential Architecture
    model = Sequential([
        # Dense input layer with ReLU activation & Dropout 0.2
        Dense(32, activation="relu", input_shape=(input_dim,), name="input_dense_layer"),
        Dropout(0.2, name="dropout_layer"),
        
        # Hidden layer with Dense ReLU & Batch Normalization
        Dense(16, activation="relu", name="hidden_dense_layer"),
        BatchNormalization(name="batch_normalization_layer"),
        
        # Output layer with Dense Sigmoid activation
        Dense(1, activation="sigmoid", name="output_sigmoid_layer")
    ])

    # Compile model with Adam optimizer and binary_crossentropy loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", "auc"]
    )

    print("\n--- ANN Model Summary ---")
    model.summary()

    # Configure EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=0
    )

    print("\n[ANN] Training Artificial Neural Network...")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0
    )

    print("[ANN] Model training completed successfully!")
    return model, history
