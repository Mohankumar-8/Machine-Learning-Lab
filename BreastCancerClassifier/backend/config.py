from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

TRAIN_CSV = BASE_DIR / "dataset" / "train.csv"
TEST_CSV = BASE_DIR / "dataset" / "test.csv"
MODEL_PATH = BASE_DIR / "models" / "breast_cancer_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"
OUTPUT_DIR = BASE_DIR / "outputs"
DB_PATH = ROOT_DIR / "database" / "breast_cancer.db"

CLASSES = ["Malignant", "Benign"]   # sklearn WDBC encoding: 0 = malignant, 1 = benign
TOP_K_FEATURES = 15                 # features kept after selection
RANDOM_STATE = 42
