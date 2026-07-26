"""
Main Pipeline Entry Point for Heart Disease Risk Predictive Modeling.

Orchestrates data loading, exploratory data analysis (EDA), 
preprocessing with StandardScaler & SMOTE, hyperparameter tuning with GridSearchCV for Random Forest, 
Keras ANN training with EarlyStopping, and comprehensive model evaluation & visualization.
"""

import sys
import argparse
from src.data_loader import load_or_generate_data
from src.eda import perform_eda
from src.preprocessing import prepare_data
from src.models import train_random_forest, build_and_train_ann
from src.evaluation import evaluate_models


def run_pipeline(data_path: str = None, n_samples: int = 1000, output_dir: str = "plots") -> None:
    """
    Executes the full end-to-end predictive modeling pipeline.

    Parameters:
        data_path (str, optional): Path to input dataset CSV.
        n_samples (int): Number of synthetic samples if dataset is generated.
        output_dir (str): Folder path for visual plot exports.
    """
    print("\n=======================================================")
    print("  PREDICTIVE MODELING FOR HEART DISEASE RISK PIPELINE  ")
    print("=======================================================")

    # Step 1: Data Loading & Synthetic Generation
    df = load_or_generate_data(filepath=data_path, n_samples=n_samples, random_state=42)

    # Step 2: Exploratory Data Analysis (EDA)
    perform_eda(df, output_dir=output_dir)

    # Step 3: Preprocessing Pipeline (Split, StandardScaler, SMOTE)
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(
        df, target_col="target", test_size=0.2, random_state=42, use_smote=True
    )

    # Step 4: Model Building & Hyperparameter Tuning
    # Model A: Random Forest (GridSearchCV)
    best_rf, best_params = train_random_forest(X_train, y_train, random_state=42, cv=5)

    # Model B: Artificial Neural Network (Keras ANN)
    input_dim = X_train.shape[1]
    ann_model, history = build_and_train_ann(
        X_train, y_train, input_dim=input_dim, epochs=100, batch_size=32, validation_split=0.2, random_state=42
    )

    # Step 5: Evaluation & Visual Comparison
    metrics_summary = evaluate_models(
        rf_model=best_rf,
        ann_model=ann_model,
        X_test=X_test,
        y_test=y_test,
        output_dir=output_dir
    )

    print("\n=======================================================")
    print("          PIPELINE EXECUTION COMPLETED SUCCESSFULLY    ")
    print("=======================================================")
    print(f"All plots and graphical visual artifacts saved under '{output_dir}/'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heart Disease Risk Predictive Modeling Pipeline")
    parser.add_argument("--data_path", type=str, default=None, help="Path to input dataset CSV file (optional)")
    parser.add_argument("--samples", type=int, default=1000, help="Number of synthetic samples if generated (default: 1000)")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory to save generated plot artifacts (default: plots)")
    
    args = parser.parse_args()
    run_pipeline(data_path=args.data_path, n_samples=args.samples, output_dir=args.output_dir)
