from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2

REGRESSION_FEATURES = [
    "age",
    "sex",
    "bmi",
    "bp",
    "s1",
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
]

CLASSIFICATION_FEATURES = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst concave points",
]

REGRESSION_TARGET = "disease_progression"
CLASSIFICATION_TARGET = "diagnosis"
