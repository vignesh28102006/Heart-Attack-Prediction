# CardioSense | Heart Disease Prediction System

An end-to-end Machine Learning web application designed to predict cardiovascular disease risk based on clinical and exercise diagnostic indicators. Built from scratch with Python, Scikit-learn, XGBoost, PostgreSQL, SQLAlchemy, and Streamlit.

---

## 1. Project Overview & Clinical Context

In cardiovascular medicine, missing an active disease case (**False Negative**) carries severe consequences, potentially leading to untreated myocardial infarction or sudden cardiac death. Conversely, a **False Positive** merely prompts confirmatory non-invasive testing (such as stress echocardiography or CT coronary angiography). 

For this reason, this system optimizes decision thresholds and model selection for **Recall (Sensitivity)** as the primary clinical metric, while sustaining strong ROC-AUC, Precision, and F1-score.

### Project Stack:
- **Core ML:** Python 3.12, Pandas, NumPy, Scikit-learn, XGBoost
- **Visualization:** Matplotlib, Seaborn
- **Database & ORM:** PostgreSQL, SQLAlchemy (with automated local SQLite fallback)
- **Web Interface:** Streamlit (Custom clinical glassmorphism design)
- **Testing:** Pytest

---

## 2. Dataset Information

The system utilizes the canonical **UCI Cleveland Heart Disease Dataset**:
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) / [Kaggle Mirror](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- **Instances:** 303 patient records
- **Features:** 13 clinical predictors (`age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`)
- **Target:** Binary diagnosis (`0`: Healthy / No Disease, `1`: Heart Disease Present)

---

## 3. Real Model Evaluation & Comparison Results

The three required classifiers were trained on a stratified 80% training split (with 5-fold Stratified Cross-Validation) and evaluated on the 20% holdout test split (61 patients: 28 diseased, 33 healthy).

### Test Set Performance (Genuine Results — Zero Fabrication)

| Model | Recall (Primary) ⭐ | False Negatives (FN) | True Positives (TP) | F1-Score | ROC-AUC | Accuracy | Precision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **XGBoost Classifier** (Champion) | **96.43%** | **1** | **27** | **0.9000** | **0.9589** | **90.16%** | **84.38%** |
| **Logistic Regression** (Baseline) | 92.86% | 2 | 26 | 0.8667 | 0.9665 | 86.89% | 81.25% |
| **Random Forest Classifier** | 89.29% | 3 | 25 | 0.8621 | 0.9470 | 86.89% | 83.33% |

> [!NOTE]
> **Key Finding:** **XGBoost Classifier** achieved the highest clinical Recall of **96.43%**, identifying 27 out of 28 diseased patients and yielding only **1 false negative**.

### Confusion Matrices Summary:
- **XGBoost:** TP = 27, FP = 5, TN = 28, FN = 1
- **Logistic Regression:** TP = 26, FP = 6, TN = 27, FN = 2
- **Random Forest:** TP = 25, FP = 5, TN = 28, FN = 3

---

## 4. Repository Structure

```text
heart-disease-prediction/
├── .env.example                 # Template for PostgreSQL and App settings
├── .gitignore                   # Ignore .env, models/*.joblib, __pycache__, .venv
├── README.md                    # Project documentation & runbook
├── requirements.txt             # Locked Python dependencies
├── config.py                    # Global path and DB configuration constants
├── data/
│   ├── raw/                     # Original unaltered UCI dataset CSV (heart.csv)
│   ├── processed/               # Cleaned train.csv and test.csv
│   └── eda_plots/               # 5 generated high-res EDA figures
├── models/                      # Serialized model artifacts and evaluation metrics
│   ├── logistic_regression.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── preprocessor.joblib      # ColumnTransformer with imputer & scaler
│   ├── evaluation_metrics.json  # Comprehensive metric scores for all 3 models
│   ├── model_metadata.json      # Metadata tracking training time & features
│   ├── confusion_matrices.png   # 3-panel confusion matrix figure
│   ├── roc_curves.png           # Multi-model ROC comparison
│   └── metrics_comparison.png   # Metric benchmark bar chart
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Ingestion, schema validation, data downloading
│   ├── preprocessor.py          # Leak-free ColumnTransformer pipeline
│   ├── train.py                 # Training pipeline for LR, RF, and XGBoost
│   ├── evaluate.py              # Metric calculation & diagnostic plot generation
│   ├── database.py              # PostgreSQL engine, SQLAlchemy models, CRUD operations
│   └── eda_generator.py         # Static EDA figure generator
├── app/
│   ├── __init__.py
│   ├── main.py                  # Streamlit application entry point
│   ├── components/
│   │   ├── charts.py            # Streamlit metric cards and risk gauge
│   │   └── forms.py             # 2-column patient diagnostic input form
│   └── views/
│       ├── overview.py          # Dataset info, statistics, variable distributions
│       ├── eda.py               # Correlation heatmaps, pairplots, feature boxplots
│       ├── model_comparison.py  # Model performance tables, Recall focus, ROC curves
│       ├── predict.py           # Interactive clinical risk assessment & probability
│       └── history.py           # PostgreSQL historical inferences log with filters
└── tests/
    ├── test_data.py             # Schema & ingestion validation tests
    ├── test_preprocessor.py     # Pipeline transformation & leak prevention tests
    ├── test_models.py           # Training, inference, and serialization tests
    └── test_database.py         # SQLAlchemy ORM and DB transaction tests
```

---

## 5. Installation & Setup

### 5.1 Clone & Virtual Environment
```bash
# Clone the repository
git clone <repo-url>
cd heart-disease-prediction

# Create and activate virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 5.2 Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Inside `.env`, configure your PostgreSQL credentials:
```ini
DB_HOST=localhost
DB_PORT=5432
DB_NAME=heart_disease_db
DB_USER=postgres
DB_PASSWORD=postgres
SQL_ECHO=False
```
*(If PostgreSQL is not running locally, the application automatically falls back to a local SQLite database at `data/heart_disease_fallback.db` without crashing).*

---

## 6. End-to-End Pipeline Execution

Execute the modular pipeline in sequence:

```bash
# 1. Download and validate authentic UCI Cleveland dataset
python -m src.data_loader

# 2. Fit leak-free preprocessor and generate train/test splits
python -m src.preprocessor

# 3. Train Logistic Regression, Random Forest, and XGBoost (with 5-fold CV)
python -m src.train

# 4. Evaluate models on test set, calculate Recall, and save diagnostic plots
python -m src.evaluate

# 5. Generate high-resolution Exploratory Data Analysis figures
python -m src.eda_generator

# 6. Initialize database and test connection
python -m src.database
```

---

## 7. Running the Streamlit Web Application

Launch the interactive clinical dashboard:

```bash
streamlit run app/main.py
```

Open your browser to `http://localhost:8501`.

### Application Features:
1. **🩺 Patient Risk Predictor:** Live patient risk assessment using any of the 3 trained models (defaults to XGBoost), interactive probability risk gauge (Low / Moderate / High Risk), and automatic persistence to the PostgreSQL audit log.
2. **⚖️ Model & Recall Comparison:** Deep-dive into Recall, Precision, F1, Accuracy, and ROC-AUC, complete with side-by-side Confusion Matrices and combined ROC Curves.
3. **📊 Exploratory Data Analysis:** High-resolution interactive visual insights covering correlation matrices, chest pain distributions, and chronotropic stress testing.
4. **📋 Dataset Overview:** Raw cohort explorer with interactive search filters and full clinical attribute definitions.
5. **📜 Prediction Audit History:** Full database audit trail of past patient evaluations with risk level filters, CSV export, and record clearing.

---

## 8. Automated Testing

Run the automated Pytest test suite:

```bash
pytest -v
```

All 11 unit and integration tests verify:
- Data schema, column names, and binary target consistency
- Preprocessing transformation output shapes and absence of NaNs
- Model artifact serialization, inference validity, and probability bounds
- Isolated in-memory SQLAlchemy database operations and history retrieval
