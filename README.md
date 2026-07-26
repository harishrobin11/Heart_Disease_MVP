# Predictive Modeling for Heart Disease Risk

A modular Python machine learning and deep learning project for predicting heart disease risk using **Scikit-Learn**, **TensorFlow/Keras**, **Pandas**, **NumPy**, **Imbalanced-Learn (SMOTE)**, and **Matplotlib/Seaborn**.

---

## Key Features

1. **Flexible Data Pipeline**:
   - Downloads/loads the UCI Heart Disease dataset or synthetically generates a 1,000-row dataset with 14 clinical features (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`).
2. **Exploratory Data Analysis (EDA)**:
   - Summary statistics, missing value audit, class distribution plot, and feature correlation heatmap.
3. **Preprocessing Pipeline**:
   - Stratified train/test split (80/20 ratio).
   - Continuous feature normalization using `StandardScaler`.
   - Oversampling class imbalance resolution using `SMOTE`.
4. **Machine Learning & Deep Learning Models**:
   - **Random Forest Classifier**: Hyperparameter-tuned with 5-fold `GridSearchCV` (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`).
   - **Keras Artificial Neural Network (ANN)**: Dense input layer with ReLU & Dropout (0.2), hidden layer with ReLU & BatchNormalization, binary Sigmoid output, compiled with Adam optimizer and `binary_crossentropy` loss, featuring `EarlyStopping`.
5. **Evaluation & Visualization**:
   - Precision, Recall, F1-score Classification Reports.
   - Side-by-side Confusion Matrices.
   - Overlaid ROC-AUC Curve comparison.

---

## Project Structure

```
Heart_Disease_MVP/
├── src/
│   ├── __init__.py          # Package exports
│   ├── data_loader.py       # Dataset fetcher & domain synthetic data generator
│   ├── eda.py               # Exploratory statistical analysis & heatmaps
│   ├── preprocessing.py     # Train/test split, StandardScaler, SMOTE balancing
│   ├── models.py            # Random Forest (GridSearchCV) & Keras ANN architectures
│   └── evaluation.py        # Metrics, confusion matrices, ROC-AUC comparison curves
├── plots/                   # Saved visualization PNG artifacts
├── main.py                  # End-to-end execution script
├── requirements.txt         # Dependencies list
└── README.md                # Documentation
```

---

## Setup & Installation

1. **Clone/Navigate to Project Folder**:
   ```bash
   cd Heart_Disease_MVP
   ```

2. **Create & Activate Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Execution

Run the complete pipeline using the virtual environment interpreter:

```bash
python main.py
```

### Custom Options:
- Specify custom CSV dataset path:
  ```bash
  python main.py --data_path path/to/your/heart.csv
  ```
- Adjust synthetic sample size:
  ```bash
  python main.py --samples 2000
  ```
- Change plot export directory:
  ```bash
  python main.py --output_dir my_plots
  ```

---

## Output Artifacts

After execution, visual artifacts are generated in the `plots/` folder:
- `plots/target_distribution.png`: Bar plot showing target class counts.
- `plots/correlation_heatmap.png`: Feature correlation matrix.
- `plots/confusion_matrices.png`: Side-by-side Random Forest & ANN confusion matrices.
- `plots/roc_auc_comparison.png`: Overlaid ROC-AUC performance comparison curves.
