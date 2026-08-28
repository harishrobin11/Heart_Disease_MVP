"""
PDF Generator for Heart Disease MVP Technical Interview & Architecture Guide.
Generates 'Heart_Disease_MVP_Interview_Prep_Guide.pdf' with comprehensive 
architecture walkthroughs, code explanations, and 54 technical Q&As.
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to add headers and 'Page X of Y' footers dynamically.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        
        # Suppress headers/footers on cover page (page 1)
        if self._pageNumber > 1:
            # Header
            self.setFont("Helvetica-Bold", 8)
            self.setFillColor(colors.HexColor("#1E293B"))
            self.drawString(54, 750, "HEART DISEASE RISK MVP — TECHNICAL INTERVIEW & ARCHITECTURE GUIDE")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
            # Footer
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#64748B"))
            self.drawString(54, 36, "CONFIDENTIAL — PREPARED FOR TECHNICAL INTERVIEW MASTERY")
            page_text = f"Page {self._pageNumber} of {page_count}"
            self.drawRightString(558, 36, page_text)
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, 48, 558, 48)
            
        self.restoreState()


def build_pdf():
    pdf_filename = "Heart_Disease_MVP_Interview_Prep_Guide.pdf"
    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY = colors.HexColor("#1E293B")      # Deep Slate Blue/Navy
    SECONDARY = colors.HexColor("#0D9488")    # Medical Teal
    ACCENT = colors.HexColor("#2563EB")       # Royal Blue
    TEXT_DARK = colors.HexColor("#334155")    # Charcoal Dark Slate
    BG_LIGHT = colors.HexColor("#F8FAFC")     # Soft Off-white
    BORDER_COLOR = colors.HexColor("#E2E8F0") # Border Gray
    CALLOUT_BG = colors.HexColor("#F0FDF4")   # Soft Mint Green Light Box
    CALLOUT_BORDER = colors.HexColor("#16A34A")

    # Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=30,
        textColor=PRIMARY,
        spaceAfter=10
    )

    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=SECONDARY,
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=PRIMARY,
        spaceBefore=16,
        spaceAfter=10,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=SECONDARY,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=TEXT_DARK,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )

    q_title_style = ParagraphStyle(
        'QTitle_Style',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=PRIMARY,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    a_text_style = ParagraphStyle(
        'AText_Style',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=TEXT_DARK,
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'Code_Style',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=4,
        spaceAfter=6
    )

    story = []

    # ==========================================
    # COVER / HEADER BANNER
    # ==========================================
    story.append(Spacer(1, 10))
    story.append(Paragraph("Heart Disease Risk Prediction MVP", title_style))
    story.append(Paragraph("Complete Technical Architecture Walkthrough & Exhaustive Interview Preparation Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=SECONDARY, spaceBefore=0, spaceAfter=15))

    # Executive Overview Box
    exec_summary_text = """
    <b>EXECUTIVE SUMMARY & INTERVIEW CHEAT SHEET</b><br/>
    This comprehensive reference manual is engineered to prepare developers, data scientists, and MLOps engineers for deep technical interviews on the <b>Heart Disease Risk Prediction MVP</b> project.<br/><br/>
    <b>Project Core:</b> An end-to-end medical risk stratification machine learning & deep learning system featuring data ingestion, SMOTE imbalance resolution, Random Forest GridSearchCV hyperparameter tuning, Keras Sequential Artificial Neural Networks (ANN), FastAPI REST serving, and a modern Glassmorphic Web Dashboard.<br/><br/>
    <b>Key Highlights Covered in this Guide:</b><br/>
    &bull; Complete system architecture & multi-layered module breakdown.<br/>
    &bull; In-depth mathematical & technical explanations for SMOTE, StandardScaler, ROC-AUC, and Keras ANN design.<br/>
    &bull; <b>54 Categorized Technical Interview Questions & Answers</b> spanning ML theory, deep learning, MLOps, FastAPI, clinical domain, and enterprise system design.<br/>
    &bull; Executive pitches, STAR method stories, and code debugging insights.
    """
    exec_table = Table(
        [[Paragraph(exec_summary_text, body_style)]],
        colWidths=[504]
    )
    exec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), BG_LIGHT),
        ('BOX', (0, 0), (-1, -1), 1, BORDER_COLOR),
        ('PADDING', (0, 0), (-1, -1), 10),
        ('LINELEFT', (0, 0), (-1, -1), 4, SECONDARY),
    ]))
    story.append(exec_table)
    story.append(Spacer(1, 15))

    # ==========================================
    # SECTION 1: ARCHITECTURE OVERVIEW
    # ==========================================
    story.append(Paragraph("1. System Architecture & Technical Stack Blueprint", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=0, spaceAfter=10))

    arch_text = """
    The system follows a strict modular MLOps architecture designed for high scalability, zero data leakage, and production-grade serving. The data flows sequentially from raw clinical feature acquisition to serialized model serving.
    """
    story.append(Paragraph(arch_text, body_style))

    # Architecture Flow Diagram Table
    arch_flow_data = [
        [Paragraph("<b>Pipeline Stage</b>", h2_style), Paragraph("<b>Module / Tech Stack</b>", h2_style), Paragraph("<b>Key Responsibilities & Operations</b>", h2_style)],
        [
            Paragraph("<b>1. Ingestion</b>", body_style),
            Paragraph("<code>src/data_loader.py</code><br/>Pandas, NumPy", body_style),
            Paragraph("Loads UCI Heart Disease dataset or synthetically generates 1,000 domain-realistic clinical rows using log-odds distribution equations.", body_style)
        ],
        [
            Paragraph("<b>2. EDA</b>", body_style),
            Paragraph("<code>src/eda.py</code><br/>Matplotlib, Seaborn", body_style),
            Paragraph("Performs statistical summaries, missing data audits, target balance checks, and feature correlation matrix heatmap generation.", body_style)
        ],
        [
            Paragraph("<b>3. Preprocessing</b>", body_style),
            Paragraph("<code>src/preprocessing.py</code><br/>Scikit-Learn, SMOTE", body_style),
            Paragraph("Executes stratified 80/20 train/test split, fits <code>StandardScaler</code> on continuous features, serializes <code>scaler.pkl</code>, and balances training set via SMOTE.", body_style)
        ],
        [
            Paragraph("<b>4. Training</b>", body_style),
            Paragraph("<code>src/train.py</code>, <code>src/models.py</code><br/>Scikit-Learn, Keras/TF", body_style),
            Paragraph("<b>Model A:</b> Random Forest tuned via 5-fold <code>GridSearchCV</code>.<br/><b>Model B:</b> Keras ANN (Dense 32 ReLU -> Dropout 0.2 -> Dense 16 ReLU -> Batch Normalization -> Sigmoid Output) with <code>EarlyStopping</code>.<br/>Serializes <code>random_forest_model.pkl</code> and <code>ann_model.h5</code>.", body_style)
        ],
        [
            Paragraph("<b>5. Evaluation</b>", body_style),
            Paragraph("<code>src/evaluate.py</code>, <code>src/evaluation.py</code><br/>Scikit-Learn, Seaborn", body_style),
            Paragraph("Computes classification reports (Precision, Recall, F1), generates side-by-side Confusion Matrices, and exports overlaid ROC-AUC performance curves.", body_style)
        ],
        [
            Paragraph("<b>6. API Serving</b>", body_style),
            Paragraph("<code>app.py</code><br/>FastAPI, Uvicorn, Pydantic", body_style),
            Paragraph("Loads serialized artifacts at startup. Exposes REST <code>/predict</code> POST endpoint with Pydantic payload validation and returns risk level, probability score %, and clinical advisory.", body_style)
        ],
        [
            Paragraph("<b>7. Web Dashboard</b>", body_style),
            Paragraph("<code>static/</code><br/>HTML5, CSS3, Vanilla JS", body_style),
            Paragraph("Interactive Glassmorphic calculator interface with dynamic SVG risk gauge meter, real-time prediction AJAX calls, and visual plot gallery viewer.", body_style)
        ]
    ]

    arch_table = Table(arch_flow_data, colWidths=[90, 114, 300])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PRIMARY),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('PADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, BG_LIGHT]),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 15))

    # ==========================================
    # SECTION 2: STEP-BY-STEP BUILD WALKTHROUGH
    # ==========================================
    story.append(Paragraph("2. Deep-Dive Technical Implementation Walkthrough", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=0, spaceAfter=10))

    walkthrough_items = [
        ("A. Data Ingestion & Synthetic Log-Odds Engine", 
         "The pipeline supports both real UCI Heart Disease CSV files and dynamic synthetic data generation. "
         "Synthetic records are created using clinical probability distributions (e.g., normal distribution for resting BP and cholesterol, binomial for exercise angina). "
         "Target heart disease labels are determined via a medical log-odds formula: <code>logit = -2.5 + 0.03*(age-50) + 0.6*(cp>0) + 0.015*(trestbps-120) - 0.03*(thalach-150) + 0.8*exang + 0.5*oldpeak + 0.6*ca</code>, "
         "which is passed through a Sigmoid activation <code>1/(1+e^-logit)</code> to yield realistic non-linear risk outcomes."),
        
        ("B. Data Leakage Prevention & Preprocessing Pipeline",
         "A critical requirement in clinical ML is avoiding data leakage. Data splitting (80/20) is strictly stratified by the target column prior to any scaling or resampling. "
         "The <code>StandardScaler</code> is fitted ONLY on the training features (continuous variables: <code>age</code>, <code>trestbps</code>, <code>chol</code>, <code>thalach</code>, <code>oldpeak</code>) and saved to <code>artifacts/scaler.pkl</code> using <code>joblib</code>. "
         "SMOTE (Synthetic Minority Over-sampling Technique) is applied exclusively to the training set to resolve class imbalance, ensuring the test set remains completely pristine and representative of real-world clinical distributions."),

        ("C. Model A: Hyperparameter-Tuned Random Forest",
         "Random Forest Classifier was selected for its robustness on tabular clinical data and intrinsic immunity to monotonic feature scaling. "
         "Hyperparameters are optimized using 5-Fold <code>GridSearchCV</code> across <code>n_estimators</code> ([50, 100]), <code>max_depth</code> ([4, 8, None]), <code>min_samples_split</code> ([2, 5]), and <code>min_samples_leaf</code> ([1, 2]), with ROC-AUC as the scoring metric. "
         "To prevent macOS fork deadlocks when running alongside TensorFlow C++ runtimes, <code>n_jobs=1</code> is explicitly enforced."),

        ("D. Model B: Keras Artificial Neural Network (ANN)",
         "Built with TensorFlow/Keras Sequential API for deep tabular pattern recognition. "
         "Architecture: Dense input layer (32 units, ReLU activation, Dropout 0.2), Hidden layer (16 units, ReLU, Batch Normalization), Output layer (1 unit, Sigmoid activation for binary classification). "
         "Compiled with <code>Adam</code> optimizer (learning rate 0.001) and <code>binary_crossentropy</code> loss. "
         "Features an <code>EarlyStopping</code> callback monitoring <code>val_loss</code> with <code>patience=10</code> and <code>restore_best_weights=True</code> to prevent overfitting."),

        ("E. Comprehensive Model Evaluation & Plot Generation",
         "Models are evaluated on the scaled test set. Metrics computed include Accuracy, Precision, Recall (Sensitivity), F1-Score, and ROC-AUC. "
         "Visualization artifacts exported to <code>plots/</code> include side-by-side Confusion Matrices and overlaid ROC-AUC comparison curves (comparing Random Forest vs. Keras ANN against a random baseline classifier)."),

        ("F. Production REST API & Interactive Web UI",
         "Built with FastAPI for asynchronous performance and automatic OpenAPI documentation. "
         "Implements a startup event handler that preloads <code>scaler.pkl</code> and <code>random_forest_model.pkl</code> into global memory. "
         "Pydantic <code>PatientPayload</code> enforces clinical value boundary constraints (e.g. blood pressure 50-250 mmHg). "
         "The <code>/predict</code> POST endpoint converts input JSON to scaled numpy arrays, executes inference, calculates probability %, determines risk class ('High Risk' vs 'Low Risk'), and returns a formatted medical advisory. "
         "The frontend is a glassmorphic dashboard styled with CSS custom variables, backdrop filters, and responsive layout grid.")
    ]

    for title, desc in walkthrough_items:
        story.append(Paragraph(title, h2_style))
        story.append(Paragraph(desc, body_style))

    story.append(Spacer(1, 10))

    # ==========================================
    # SECTION 3: 54 INTERVIEW Q&AS CATALOG
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("3. Exhaustive 54 Interview Questions & Answers Catalog", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=0, spaceAfter=15))

    qna_categories = [
        ("Category A: Machine Learning Theory, Statistics & Preprocessing (Q1 - Q12)", [
            ("Q1: Why did you choose Random Forest and Neural Networks for tabular clinical data prediction?",
             "Random Forest is an ensemble tree method that excels on tabular datasets of moderate size (1,000 rows), handling non-linear interactions and mixed feature types without requiring strict feature scaling. "
             "Artificial Neural Networks (ANNs) were implemented alongside to capture complex high-order feature combinations via dense representation layers. Comparing both allowed us to benchmark classical ensemble learning against deep learning."),

            ("Q2: Explain how SMOTE works. Why must it be applied ONLY to the training set?",
             "SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for the minority class by interpolating between existing minority instances and their k-nearest neighbors in feature space. "
             "It MUST be applied exclusively to the training set. Applying SMOTE to the test set or before train/test splitting causes severe data leakage: synthetic test points would be created from training data neighbors, causing artificially inflated performance metrics."),

            ("Q3: What is Data Leakage and how did your architecture strictly prevent it?",
             "Data leakage occurs when information from outside the training dataset is used to train the model. In our pipeline, leakage was prevented by: "
             "1) Performing stratified train/test split BEFORE any scaling or resampling. "
             "2) Fitting <code>StandardScaler</code> ONLY on the training set (<code>X_train</code>) and using <code>scaler.transform()</code> on the test set (<code>X_test</code>). "
             "3) Applying SMOTE oversampling exclusively on training data."),

            ("Q4: Why use StandardScaler over MinMaxScaler for continuous clinical features like cholesterol and blood pressure?",
             "StandardScaler scales features to have zero mean and unit variance (z-score normalization: <code>(x - μ) / σ</code>). "
             "Clinical features like resting blood pressure (<code>trestbps</code>) and cholesterol (<code>chol</code>) approximate normal Gaussian distributions with occasional extreme outliers. "
             "MinMaxScaler squashes all values into a fixed [0, 1] range, making it highly sensitive to outliers, which compresses non-outlier data into a tiny range. StandardScaler handles Gaussian distributions and extreme clinical outliers far more robustly."),

            ("Q5: Why is Recall (Sensitivity) prioritized over Precision in heart disease prediction?",
             "In medical diagnostic screening, a False Negative (classifying a high-risk cardiac patient as 'Low Risk') can lead to delayed treatment and fatal outcomes. A False Positive (classifying a healthy person as 'High Risk') merely results in follow-up diagnostic testing. "
             "Therefore, high Recall (Sensitivity = <code>TP / (TP + FN)</code>) is critical to catch every potential cardiac patient."),

            ("Q6: How do you interpret ROC-AUC vs Precision-Recall AUC for imbalanced datasets?",
             "ROC-AUC measures the trade-off between True Positive Rate (Sensitivity) and False Positive Rate (1 - Specificity) across all decision thresholds. An AUC of 1.0 represents a perfect classifier, while 0.5 represents a random guess. "
             "However, when class imbalance is severe, ROC-AUC can present an overly optimistic view because the False Positive Rate denominator includes a large number of true negatives. In such cases, Precision-Recall AUC is more informative."),

            ("Q7: Explain Stratified K-Fold cross-validation vs standard K-Fold split.",
             "Standard K-Fold splits dataset randomly into K folds. If target classes are imbalanced (e.g. 80% low risk, 20% high risk), random splitting can lead to folds with zero or very few minority samples. "
             "Stratified K-Fold enforces that every fold maintains the exact same class ratio (80/20) as the complete dataset, ensuring stable cross-validation metrics."),

            ("Q8: How did you handle continuous vs categorical clinical features during scaling?",
             "Categorical variables (e.g., <code>sex</code>, <code>fbs</code>, <code>exang</code>) and ordinal categorical features (<code>cp</code>, <code>restecg</code>, <code>slope</code>, <code>ca</code>, <code>thal</code>) are already integer-encoded discrete values. "
             "Continuous features (<code>age</code>, <code>trestbps</code>, <code>chol</code>, <code>thalach</code>, <code>oldpeak</code>) were isolated and passed to <code>StandardScaler</code>. Discrete binary and ordinal categorical features were preserved without scaling to maintain their exact numerical integrity."),

            ("Q9: What are the mathematical assumptions behind synthetic data generation in src/data_loader.py?",
             "The synthetic generator uses clinical baseline distributions: age uniform between 29-77, resting blood pressure Gaussian <code>N(131, 17.5)</code>, cholesterol Gaussian <code>N(246, 50)</code>, and maximum heart rate linearly tied to age (<code>220 - age</code>). "
             "The binary target is calculated using a log-odds logistic regression function where chest pain type, exercise angina, ST depression, and major vessels increase log-odds of heart disease."),

            ("Q10: How do tree-based ensemble models handle non-linear feature relationships compared to neural networks?",
             "Random Forest handles non-linearity by making axis-aligned decision splits across feature thresholds, building piece-wise step function decision boundaries. "
             "Neural Networks handle non-linearity through continuous non-linear activation functions (e.g., ReLU, Sigmoid) applied to linear matrix multiplications, learning continuous curved decision boundaries in high-dimensional feature space."),

            ("Q11: What is the effect of feature correlation on Random Forest feature importance?",
             "When two features are highly correlated (e.g., <code>oldpeak</code> and <code>slope</code>), Random Forest can split on either feature arbitrarily. As a result, the importance score is split between them, making both appear less important than they actually are. "
             "We checked the correlation matrix in <code>src/eda.py</code> to audit collinearity."),

            ("Q12: How would you handle missing clinical data in a real-world production pipeline?",
             "Missing data should be handled via a domain-aware imputation pipeline. For continuous features (blood pressure, cholesterol), KNN Imputer or IterativeImputer (MICE) is preferred over simple median imputation. For categorical clinical features, missing values can be assigned a distinct category code (e.g., 'Unknown'). All imputation transformers must be fitted on training data and serialized.")
        ]),

        ("Category B: Deep Learning & Neural Network Architecture (Q13 - Q22)", [
            ("Q13: Walk me through your Keras ANN architecture. Why choose 32 and 16 units?",
             "The ANN architecture uses a funnel design suited for low-to-medium dimensional tabular data (13 features): "
             "- Input Layer: <code>Dense(32, activation='relu')</code> expands 13 features into a 32-dimensional non-linear feature space. "
             "- Regularization: <code>Dropout(0.2)</code> randomly zeros 20% of activations during training to prevent co-adaptation of weights. "
             "- Hidden Layer: <code>Dense(16, activation='relu')</code> compresses representations down to 16 abstract features. "
             "- Normalization: <code>BatchNormalization()</code> standardizes layer outputs across mini-batches. "
             "- Output Layer: <code>Dense(1, activation='sigmoid')</code> outputs scalar risk probability between 0.0 and 1.0."),

            ("Q14: Why use Sigmoid activation in the final layer instead of Softmax?",
             "Softmax is used for multi-class classification where output probabilities across all N classes must sum to 1.0. "
             "Our model performs binary classification (Heart Disease vs. Healthy). A single neuron with a Sigmoid activation <code>σ(z) = 1 / (1 + e^-z)</code> maps real-valued logits to a single probability score in range [0, 1]."),

            ("Q15: What is the role of Batch Normalization in your ANN hidden layer?",
             "Batch Normalization standardizes inputs to a layer by subtracting batch mean and dividing by batch standard deviation, then applying learnable scale (γ) and shift (β) parameters. "
             "It mitigates Internal Covariate Shift, accelerates training convergence, allows higher learning rates, and provides a mild regularizing effect."),

            ("Q16: Why add a Dropout layer (0.2) after the input layer?",
             "Dropout randomly sets 20% of neuron outputs to zero during each training forward pass. "
             "This prevents the neural network from relying excessively on any single feature or weight connection, forcing the model to learn redundant, robust feature representations and effectively preventing overfitting on small clinical datasets."),

            ("Q17: How does EarlyStopping work? Why set restore_best_weights=True?",
             "<code>EarlyStopping</code> monitors a validation metric (here <code>val_loss</code>). If validation loss fails to improve for 10 consecutive epochs (<code>patience=10</code>), training stops immediately. "
             "Setting <code>restore_best_weights=True</code> ensures that the model weights are rolled back to the exact epoch that achieved the lowest validation loss, rather than retaining the overfitted weights from the final training epoch."),

            ("Q18: Why choose binary_crossentropy loss over Mean Squared Error (MSE)?",
             "Binary Cross-Entropy loss <code>L = -[y*log(p) + (1-y)*log(1-p)]</code> is derived from maximum likelihood estimation for Bernoulli distributions. "
             "It penalizes confident incorrect predictions exponentially. MSE produces small gradient updates when Sigmoid outputs saturate near 0 or 1, causing slow training, whereas binary cross-entropy maintains strong gradient signals."),

            ("Q19: How did you select the learning rate for the Adam optimizer?",
             "The Adam optimizer was initialized with default learning rate <code>0.001</code>, which dynamically adjusts per-parameter learning rates using first-order (mean) and second-order (uncentered variance) moments of gradients. "
             "Combined with Batch Normalization and EarlyStopping, 0.001 achieved stable convergence without gradient divergence."),

            ("Q20: Why might an ANN underperform Random Forest on tabular datasets of 1,000 rows?",
             "Neural networks are parameter-dense and rely on smooth continuous representations, requiring large sample sizes to optimize decision boundaries without overfitting. "
             "Tree ensembles like Random Forest build discrete split thresholds and exhibit strong inductive bias for tabular structures, often outperforming ANNs on small-to-medium tabular data."),

            ("Q21: What is the vanishing gradient problem and how do ReLU and Batch Normalization prevent it?",
             "Vanishing gradients occur when backpropagated gradients shrink exponentially through deep layers with saturating activations like Sigmoid or Tanh, halting weight updates. "
             "ReLU has constant derivative of 1 for positive inputs, preventing gradient decay. Batch Normalization keeps layer inputs in active, non-saturating regions."),

            ("Q22: How would you systematically tune ANN hyperparameters using KerasTuner?",
             "We would define a search space covering layer depth (1 to 4 layers), unit counts (16 to 128), dropout rates (0.0 to 0.5), activation functions (ReLU, ELU, Swish), and learning rates (1e-4 to 1e-2). We would then run Bayesian Optimization via KerasTuner to find optimal performance under cross-validation.")
        ]),

        ("Category C: MLOps, Serialization & Architecture (Q23 - Q32)", [
            ("Q23: How are model artifacts serialized and saved in this project?",
             "Artifacts are serialized during training in <code>main.py</code>: "
             "1) <code>StandardScaler</code> is serialized via <code>joblib.dump(scaler, 'artifacts/scaler.pkl')</code>. "
             "2) Best Random Forest estimator is saved via <code>joblib.dump(best_rf, 'artifacts/random_forest_model.pkl')</code>. "
             "3) Keras ANN model architecture, weights, and optimizer state are saved via <code>model.save('artifacts/ann_model.h5')</code>."),

            ("Q24: What is the difference between joblib pickle serialization and Keras HDF5 format?",
             "<code>joblib</code> is optimized for fast binary serialization of arbitrary Python objects containing large numerical NumPy arrays, making it ideal for Scikit-Learn estimators and scalers. "
             "HDF5 (<code>.h5</code>) is a hierarchical data format specifically designed for storing neural network model topologies, layer configurations, weight tensors, and optimizer states in a platform-independent format."),

            ("Q25: How does FastAPI load serialized artifacts into memory at server startup?",
             "FastAPI uses the <code>@app.on_event('startup')</code> lifecycle hook. When the server launches, <code>load_artifacts()</code> is executed once: "
             "It checks if <code>scaler.pkl</code> and <code>random_forest_model.pkl</code> exist in <code>artifacts/</code> (if missing, it automatically triggers <code>main.py</code> to execute the training pipeline), loads them into global memory, and makes them instantly available for fast REST inference."),

            ("Q26: How do you detect and handle feature drift or schema drift in production?",
             "Schema drift (missing or malformed fields, changed data types) is intercepted at the API boundary using Pydantic models (<code>PatientPayload</code>), which reject non-compliant inputs with HTTP 422 error codes. "
             "Feature drift (changes in continuous input distributions over time) can be monitored by comparing incoming inference payload distributions against baseline training distributions using statistical tests like Kolmogorov-Smirnov (KS) or Population Stability Index (PSI)."),

            ("Q27: How would you automate model retraining when new patient records arrive?",
             "An automated MLOps retraining pipeline can be established using Apache Airflow or Prefect: "
             "1) Data ingestion trigger fires when new patient records land in data lake (e.g. S3 bucket). "
             "2) Retraining pipeline executes data quality checks, preprocessing, SMOTE balancing, and model tuning. "
             "3) Candidate model is evaluated against the active production model. "
             "4) If candidate model achieves superior ROC-AUC on holdout validation data, artifacts are updated and deployed via blue-green deployment."),

            ("Q28: Describe your project directory structure and why separation of concerns matters.",
             "The project isolates functional responsibilities into modular scripts inside <code>src/</code>: "
             "- <code>data_loader.py</code>: Data fetcher & domain generator. "
             "- <code>eda.py</code>: Exploratory data analysis & heatmaps. "
             "- <code>preprocessing.py</code>: Train/test split, scaling, SMOTE. "
             "- <code>models.py</code> & <code>train.py</code>: Model building & hyperparameter grid search. "
             "- <code>evaluation.py</code> & <code>evaluate.py</code>: Performance metrics & graphic exports. "
             "- <code>app.py</code>: FastAPI web server & Pydantic schemas. "
             "Separation of concerns ensures clean unit testing, modular maintenance, and zero code duplication."),

            ("Q29: Explain the macOS fork deadlock issue with TensorFlow and how setting n_jobs=1 resolved it.",
             "On macOS, Python's <code>multiprocessing</code> uses <code>fork()</code> by default. When <code>GridSearchCV(n_jobs=-1)</code> attempts to spawn child processes after TensorFlow's C++ runtime has initialized multi-threaded OpenMP/Eigen threads, the child processes inherit deadlocked mutex locks, causing the execution to hang indefinitely. "
             "Setting <code>n_jobs=1</code> forces single-threaded sequential cross-validation execution, bypassing POSIX fork deadlock constraints."),

            ("Q30: How do you version control models, code, and datasets?",
             "Code is version-controlled via Git. Datasets and serialized model binary artifacts (<code>.pkl</code>, <code>.h5</code>) should be version-controlled using DVC (Data Version Control) backed by remote storage (S3, GCS). Git tracks lightweight <code>.dvc</code> pointer files, maintaining perfect reproducibility across code and model checkpoints."),

            ("Q31: How would you monitor live inference latencies and model performance?",
             "FastAPI middleware can record request-response latencies and push metrics to Prometheus, visualized via Grafana dashboards. "
             "To monitor real-world accuracy, predictions are logged with unique patient IDs to a database. When actual clinical diagnostic ground truths become available, automated evaluation scripts compare predicted risk vs. actual clinical outcomes."),

            ("Q32: How do you ensure reproducible random seeds across NumPy, Scikit-Learn, and TensorFlow?",
             "Reproducibility is enforced by explicitly setting deterministic random seeds across all libraries at the entry point: "
             "<code>np.random.seed(42)</code>, <code>tf.random.set_seed(42)</code>, and passing <code>random_state=42</code> to Scikit-Learn estimators (<code>train_test_split</code>, <code>RandomForestClassifier</code>, <code>SMOTE</code>).")
        ]),

        ("Category D: FastAPI Backend & Web Architecture (Q33 - Q42)", [
            ("Q33: Why choose FastAPI over Flask or Django for serving ML model endpoints?",
             "FastAPI offers key technical advantages for ML model serving: "
             "1) High Performance: Built on Starlette and Pydantic, utilizing ASGI async event loops. "
             "2) Data Validation: Automatic payload parsing and type enforcement via Pydantic. "
             "3) Auto Documentation: Generates interactive Swagger UI (<code>/docs</code>) and ReDoc documentation out of the box. "
             "4) Asynchronous Endpoints: Native <code>async/await</code> support for concurrent request handling."),

            ("Q34: How does Pydantic enforce clinical feature input bounds in PatientPayload?",
             "Pydantic's <code>BaseModel</code> uses <code>Field</code> constraints: "
             "For example, <code>age: int = Field(..., ge=1, le=120)</code>, <code>trestbps: int = Field(..., ge=50, le=250)</code>, and <code>chol: int = Field(..., ge=100, le=600)</code>. "
             "If an incoming JSON request violates any bound (e.g. <code>chol: 9999</code>), Pydantic automatically rejects the payload before reaching inference code, returning an HTTP 422 Unprocessable Entity response with structured field error descriptions."),

            ("Q35: What happens if an API client sends invalid or out-of-range clinical inputs?",
             "FastAPI intercepts the request at the Pydantic parsing layer and returns an HTTP 422 error response detailing the exact validation failure (e.g., field name, constraint violated, expected vs provided value). "
             "If an unexpected unhandled exception occurs during inference, FastAPI's exception handler catches it and returns HTTP 500 Internal Server Error without crashing the API process."),

            ("Q36: How is CORS configured and why is it necessary for web frontend interaction?",
             "CORS (Cross-Origin Resource Sharing) is enabled via FastAPI's <code>CORSMiddleware</code>: "
             "<code>app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])</code>. "
             "This permits browser frontends hosted on different domains or ports (e.g. static HTML dashboard or React app) to make HTTP POST requests to the API without being blocked by browser Same-Origin Policy safety restrictions."),

            ("Q37: Explain FastAPI startup and shutdown lifecycle events.",
             "Lifecycle event handlers run code before the application starts receiving requests or during shutdown: "
             "<code>@app.on_event('startup')</code> preloads heavy model artifacts into global memory once. This ensures that individual prediction requests do not incur disk reading overhead, providing sub-10ms response latencies."),

            ("Q38: How does static file mounting work in FastAPI?",
             "Static files (HTML, CSS, JS) are mounted via <code>app.mount('/static', StaticFiles(directory='static'), name='static')</code>. "
             "The root route <code>GET /</code> returns <code>FileResponse('static/index.html')</code>, serving the full web dashboard directly from the FastAPI application without needing a separate web server like Nginx during local development."),

            ("Q39: How would you handle high concurrent traffic on the /predict endpoint?",
             "1) Production ASGI Server: Run FastAPI with Uvicorn workers managed by Gunicorn (e.g. <code>gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app</code>). "
             "2) Horizontal Scaling: Deploy containerized instances across a Kubernetes cluster behind an AWS Application Load Balancer. "
             "3) Model Inference Optimization: Export model to ONNX or TensorRT format for high-throughput batch execution."),

            ("Q40: What HTTP status codes does your API return and what do they indicate?",
             "- HTTP 200 OK: Prediction executed successfully, returning risk classification JSON. "
             "- HTTP 400 Bad Request: Malformed JSON payload structure. "
             "- HTTP 422 Unprocessable Entity: Pydantic validation failure for out-of-bounds clinical fields. "
             "- HTTP 503 Service Unavailable: Model artifacts missing or failed to initialize during server startup."),

            ("Q41: How would you secure this prediction API in an enterprise production deployment?",
             "1) Authentication & Authorization: Require OAuth2 JWT bearer tokens or API Keys via FastAPI dependency injection (<code>Security(api_key_header)</code>). "
             "2) Rate Limiting: Apply Redis-backed rate limiting (e.g., max 100 requests/minute per client). "
             "3) TLS Encryption: Enforce HTTPS encryption for all in-transit patient payload traffic."),

            ("Q42: Explain the structure of the JSON payload returned by the /predict endpoint.",
             "The endpoint returns a JSON response containing: "
             "<code>{'status': 'success', 'prediction': 'High Risk', 'probability': 0.8452, 'risk_score_pct': '84.5%', 'clinical_advisory': 'Immediate cardiology consultation and diagnostic evaluation recommended.', 'patient_features': {...}}</code>.")
        ]),

        ("Category E: Clinical Domain Knowledge & Feature Interpretation (Q43 - Q48)", [
            ("Q43: What are the 13 clinical features in the UCI dataset and their medical significance?",
             "1. <code>age</code>: Patient age in years.<br/>"
             "2. <code>sex</code>: Gender (0 = Female, 1 = Male; males statistically show earlier CAD incidence).<br/>"
             "3. <code>cp</code>: Chest Pain Type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic).<br/>"
             "4. <code>trestbps</code>: Resting Blood Pressure in mm Hg (hypertension >130 mm Hg increases vascular strain).<br/>"
             "5. <code>chol</code>: Serum Cholesterol in mg/dl (hypercholesterolemia >200 mg/dl causes plaque buildup).<br/>"
             "6. <code>fbs</code>: Fasting Blood Sugar > 120 mg/dl (indicator of diabetic cardiovascular risk).<br/>"
             "7. <code>restecg</code>: Resting ECG (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy).<br/>"
             "8. <code>thalach</code>: Maximum Heart Rate Achieved (lower peak exercise heart rate indicates impaired cardiac reserve).<br/>"
             "9. <code>exang</code>: Exercise Induced Angina (1 = Yes, indicates coronary ischemia under stress).<br/>"
             "10. <code>oldpeak</code>: ST Depression Induced by Exercise Relative to Rest (key ECG marker for myocardial ischemia).<br/>"
             "11. <code>slope</code>: Slope of Peak Exercise ST Segment (0: upsloping, 1: flat, 2: downsloping; flat/downsloping indicates severe risk).<br/>"
             "12. <code>ca</code>: Number of Major Vessels Colored by Fluoroscopy (0-4; higher count indicates extensive coronary artery blockage).<br/>"
             "13. <code>thal</code>: Thalium Stress Test (1: fixed defect, 2: reversible defect, 3: normal)."),

            ("Q44: What is oldpeak (ST depression) and slope in an ECG test?",
             "During cardiac exercise stress testing, the ST segment of an ECG waveform reflects ventricular repolarization. "
             "<code>oldpeak</code> measures the mm of ST segment depression at peak exercise relative to rest. "
             "A horizontal (flat) or downsloping ST segment (<code>slope</code>) indicates subendocardial ischemia (insufficient blood flow to the heart muscle)."),

            ("Q45: What does thal (Thalium stress test) measure?",
             "A Thalium stress test evaluates blood perfusion in cardiac muscle tissue using radioactive thalium tracers. "
             "A 'fixed defect' indicates prior permanent heart tissue damage (myocardial infarction), while a 'reversible defect' indicates temporary ischemia caused by coronary artery obstruction under stress."),

            ("Q46: How does chest pain type (cp) correlate with heart disease risk?",
             "Surprisingly, patients exhibiting typical angina (chest pressure radiating to arm/jaw under exertion) often seek immediate care, whereas asymptomatic or atypical presentation (<code>cp=3</code>) can lead to silent ischemia and late-stage presentation. "
             "In our non-linear logistic synthetic generator and tree models, chest pain type is a dominant predictive feature."),

            ("Q47: How is the risk score percentage translated into an actionable clinical advisory?",
             "The raw model output probability (0.0 to 1.0) is converted to a percentage <code>probability * 100</code>. "
             "If probability >= 0.5, label is 'High Risk' and advisory recommends immediate cardiology referral, stress testing, and coronary angiography. If < 0.5, label is 'Low Risk' and advisory recommends routine annual wellness monitoring."),

            ("Q48: How would you explain model prediction decisions to a medical professional?",
             "Black-box predictions are unacceptable in healthcare. We would utilize SHAP (SHapley Additive exPlanations) or LIME to break down feature contributions for individual patient predictions (e.g., 'Patient risk is 82% primarily due to high fluoroscopy vessel count ca=3 (+35%) and ST depression oldpeak=2.6 (+25%)').")
        ]),

        ("Category F: Enterprise System Design & Scaling (Q49 - Q54)", [
            ("Q49: How would you containerize this Heart Disease MVP project using Docker?",
             "We would construct a multi-stage <code>Dockerfile</code>: "
             "Base stage: <code>python:3.10-slim</code>. Workdir <code>/app</code>. Copy <code>requirements.txt</code> and run <code>pip install --no-cache-dir -r requirements.txt</code>. "
             "Copy <code>src/</code>, <code>artifacts/</code>, <code>static/</code>, and <code>app.py</code>. "
             "Expose port 8000. Entrypoint command: <code>uvicorn app:app --host 0.0.0.0 --port 8000</code>."),

            ("Q50: How would you deploy this containerized service to AWS?",
             "1) Container Registry: Push Docker image to AWS Elastic Container Registry (ECR). "
             "2) Container Orchestration: Deploy image onto AWS Elastic Container Service (ECS) with Fargate launch type (serverless containers). "
             "3) Load Balancing & Networking: Place ECS task behind AWS Application Load Balancer (ALB) across multiple Availability Zones with auto-scaling rules based on CPU/RAM utilization."),

            ("Q51: How would you set up a CI/CD pipeline for this repository using GitHub Actions?",
             "Create <code>.github/workflows/deploy.yml</code>: "
             "1) On git push to <code>main</code>, launch GitHub runner. "
             "2) Set up Python environment and run unit tests (<code>pytest</code> testing data pipeline and Pydantic validation). "
             "3) Execute <code>python main.py</code> to verify model training integration. "
             "4) Build Docker image and push to ECR. "
             "5) Trigger ECS service update for zero-downtime deployment."),

            ("Q52: How would you conduct A/B testing between Random Forest and Keras ANN in live production?",
             "Deploy an API Gateway or Service Mesh router (e.g., Istio / AWS App Mesh) in front of two microservices: Service A (Random Forest) and Service B (Keras ANN). "
             "Split live incoming clinical requests 50/50. Log predicted risk scores alongside patient IDs. Compare downstream clinical outcome agreements to evaluate real-world efficacy."),

            ("Q53: How would you ensure HIPAA / GDPR compliance for patient medical data?",
             "1) Anonymization: Strip all PHI (Protected Health Information like Patient Name, SSN, DOB, Address) prior to ingestion. "
             "2) Encryption: Enforce TLS 1.3 for data in transit and AES-256 encryption for data at rest (S3 bucket encryption / KMS). "
             "3) Access Control: Apply strict Role-Based Access Control (RBAC) and audit logging for all database queries."),

            ("Q54: How would you scale the system to process batch inference for 100,000 patient records?",
             "For high-volume offline batch prediction, REST endpoints are inefficient. "
             "We would deploy PySpark or AWS EMR to distribute the serialized <code>scaler.pkl</code> and <code>random_forest_model.pkl</code> across a distributed cluster via map partitions, executing batch prediction across millions of patient records in parallel.")
        ])
    ]

    for cat_title, qna_list in qna_categories:
        story.append(Paragraph(cat_title, h1_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=SECONDARY, spaceBefore=0, spaceAfter=8))
        
        for q_text, a_text in qna_list:
            qna_block = [
                Paragraph(f"<b>{q_text}</b>", q_title_style),
                Paragraph(a_text, a_text_style)
            ]
            story.append(KeepTogether(qna_block))
            story.append(Spacer(1, 2))

    # ==========================================
    # SECTION 4: INTERVIEW PITCH & STAR STORIES
    # ==========================================
    story.append(PageBreak())
    story.append(Paragraph("4. Interview Pitch & Behavioral STAR Stories Strategy", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=0, spaceAfter=15))

    pitches = [
        ("30-Second Elevator Pitch",
         "\"I built an end-to-end Heart Disease Risk Prediction system that bridges machine learning, deep learning, and production API serving. "
         "Using clinical features from the UCI dataset, I engineered a robust MLOps pipeline with SMOTE class balancing, Random Forest GridSearchCV tuning, and a Keras Sequential ANN. "
         "I deployed the model as a FastAPI REST microservice with Pydantic payload validation and an interactive Glassmorphic web dashboard, enabling real-time cardiovascular risk assessment.\""),

        ("2-Minute Technical Project Breakdown",
         "\"The goal of this project was to create a production-ready medical risk stratification system. "
         "Starting with data ingestion, I built a flexible pipeline supporting both real UCI clinical data and dynamic synthetic data generation. "
         "To handle class imbalance without data leakage, I implemented a strict preprocessing pipeline that isolates the stratified 80/20 train/test split before fitting a StandardScaler and applying SMOTE oversampling exclusively on training data. "
         "For modeling, I evaluated two distinct paradigms: a classical Random Forest Classifier tuned via 5-fold cross-validation on ROC-AUC, and a Keras Artificial Neural Network featuring Dense ReLU layers, Dropout, Batch Normalization, and EarlyStopping. "
         "For deployment, I built a FastAPI web server that preloads serialized model artifacts into memory on startup. It exposes a POST /predict endpoint enforcing strict Pydantic clinical boundary checks and serving an interactive Glassmorphic web calculator dashboard. "
         "The final pipeline delivers high ROC-AUC performance and sub-10ms inference latencies.\""),

        ("STAR Story 1: Resolving macOS Multiprocessing Deadlocks during GridSearchCV Tuning",
         "<b>Situation:</b> During model development on macOS, executing 5-fold <code>GridSearchCV</code> with <code>n_jobs=-1</code> alongside TensorFlow caused the training script to freeze indefinitely.<br/>"
         "<b>Task:</b> Diagnose the process hanging issue and restore fast, automated hyperparameter tuning.<br/>"
         "<b>Action:</b> I analyzed process thread state stack traces and identified a POSIX <code>fork()</code> deadlock: TensorFlow's C++ runtime initializes multi-threaded OpenMP/Eigen pools, which deadlock when Python forks worker processes. I resolved this by enforcing <code>n_jobs=1</code> in GridSearchCV, enabling sequential cross-validation execution.<br/>"
         "<b>Result:</b> The training pipeline executed cleanly without process hangs, completing GridSearchCV tuning in under 15 seconds."),

        ("STAR Story 2: Strict Prevention of Data Leakage in Medical Resampling",
         "<b>Situation:</b> In initial baseline experiments, class imbalance resampling produced suspiciously perfect 99%+ test set accuracy.<br/>"
         "<b>Task:</b> Audit data preprocessing workflow for subtle data leakage.<br/>"
         "<b>Action:</b> I discovered that applying SMOTE oversampling across the full dataset prior to train/test splitting caused synthetic test points to be created from training neighbors. I refactored <code>src/preprocessing.py</code> to enforce strict pipeline ordering: 1) Stratified train/test split, 2) Fit scaler on train only, 3) Resample train set only with SMOTE.<br/>"
         "<b>Result:</b> Restored true clinical validation integrity, yielding a trustworthy ROC-AUC performance metric.")
    ]

    for title, content in pitches:
        story.append(Paragraph(title, h2_style))
        story.append(Paragraph(content, body_style))
        story.append(Spacer(1, 6))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated '{pdf_filename}'!")

if __name__ == "__main__":
    build_pdf()
