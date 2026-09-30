import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EDA_PLOTS_DIR = DATA_DIR / "eda_plots"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure runtime directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, EDA_PLOTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "heart.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "heart_processed.csv"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
LOGISTIC_REGRESSION_PATH = MODELS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_PATH = MODELS_DIR / "random_forest.joblib"
XGBOOST_PATH = MODELS_DIR / "xgboost.joblib"

# Dataset Specification
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "target"

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg on admission to hospital)",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "Resting electrocardiographic results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)",
    "ca": "Number of major vessels (0-3) colored by fluoroscopy",
    "thal": "Thallium stress result (1 = normal, 2 = fixed defect, 3 = reversible defect)",
    "target": "Presence of heart disease (0 = absence, 1 = presence)",
}

# Machine Learning Configuration
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
CV_FOLDS = 5

# Database Configuration
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "heart_disease_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
SQL_ECHO = os.getenv("SQL_ECHO", "False").lower() in ("true", "1", "yes")

POSTGRES_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
FALLBACK_SQLITE_URL = f"sqlite:///{DATA_DIR / 'heart_disease_fallback.db'}"

DATABASE_URL = os.getenv("DATABASE_URL", POSTGRES_URL)

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Models directory: {MODELS_DIR}")
    print(f"Configured Database URL: {DATABASE_URL}")
