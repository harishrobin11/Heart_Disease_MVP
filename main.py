"""
MLOps Pipeline Orchestrator for Heart Disease Risk Predictive Modeling.

Executes the end-to-end pipeline:
1. Load or Generate Raw Dataset ('data/raw/heart_disease.csv')
2. Preprocess, Scale & Balance Training Data ('artifacts/scaler.pkl')
3. Train & Serialize Random Forest & Keras ANN ('artifacts/random_forest_model.pkl', 'artifacts/ann_model.h5')
4. Evaluate Models & Save Metrics Plots ('artifacts/roc_auc_comparison.png')
"""

import os
from src.data_loader import load_or_generate_data
from src.preprocessing import run_preprocessing_pipeline
from src.train import train_random_forest, train_ann_model
from src.evaluate import evaluate_models


def main():
    raw_data_path = os.path.join("data", "raw", "heart_disease.csv")
    artifact_dir = "artifacts"

    # Ensure required directories exist dynamically
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    print("\n=======================================================")
    print("      STARTING MLOPS HEART DISEASE MODELING PIPELINE   ")
    print("=======================================================")

    # 1. Load Data / Synthetic Generation
    df = load_or_generate_data(filepath=raw_data_path, n_samples=1000)

    # 2. Preprocess & Serialize Scaler
    X_train_res, X_test_scaled, y_train_res, y_test, feature_names = run_preprocessing_pipeline(
        df, target_col="target", test_size=0.2, artifact_dir=artifact_dir
    )

    # 3. Train & Serialize Models
    rf_model, _ = train_random_forest(X_train_res, y_train_res, cv=5, artifact_dir=artifact_dir)
    
    input_dim = X_train_res.shape[1]
    ann_model, _ = train_ann_model(X_train_res, y_train_res, input_dim=input_dim, epochs=100, artifact_dir=artifact_dir)

    # 4. Evaluate & Save Graphics
    metrics = evaluate_models(rf_model, ann_model, X_test_scaled, y_test, artifact_dir=artifact_dir)

    print("\n=======================================================")
    print("    PIPELINE EXECUTED SUCCESSFULLY & ARTIFACTS SAVED    ")
    print("=======================================================")
    print(f"Serialized artifacts saved under '{artifact_dir}/':")
    print("  - artifacts/scaler.pkl")
    print("  - artifacts/random_forest_model.pkl")
    print("  - artifacts/ann_model.h5")
    print("  - artifacts/roc_auc_comparison.png")


if __name__ == "__main__":
    main()
