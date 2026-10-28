from pathlib import Path

BASE = Path(__file__).resolve().parent
MIMII_ROOT = BASE / "data" / "MIMII"
SUBF_ROOT = BASE / "data" / "Dataset"
MODEL_DIR = BASE / "models"
REPORT_DIR = BASE / "reports"
STORAGE_DIR = BASE / "storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
MODEL_PATH = MODEL_DIR / "model.joblib"
CACHE_PATH = MODEL_DIR / "features_cache.npz"
EVAL_PATH = REPORT_DIR / "evaluation.json"
EVAL_PDF_PATH = REPORT_DIR / "evaluation_report.pdf"
DB_PATH = STORAGE_DIR / "monitor.db"

MACHINES = ["fan", "pump", "slider", "valve"]
SR = 16000
SEG_SEC = 2
N_MELS = 64
N_MFCC = 20
FEATURE_VERSION = 2
CHANNELS = [0, 4]

TRAIN_SNR = 6
TEST_SNRS = [0, -6]
AUG_SNRS = [3, -3]
HOLDOUT_IDS = ["id_06"]

SUBF_SR = 10000
SUBF_CLASSES = {"Normal": 0, "Inner Race Fault": 1, "Outer Race Fault": 1}
SUBF_MAX_PER_CLASS = 300

N_TREES = 150
BOOST_ITER = 250
POOLING_MODES = ["max"]
NOVELTY_PERCENTILE = 97.5
OOD_PERCENTILE = 99
OOD_MARGIN = 1.25

WATCH_FLAG_RATIO = 0.2
CRITICAL_FLAG_RATIO = 0.8
TREND_WINDOW = 6
TREND_SLOPE = 2.0

MIN_PASSWORD = 8
LOCK_ATTEMPTS = 5
LOCK_MINUTES = 5
SAVE_UPLOADS = True
SEED = 42
