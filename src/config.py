from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
FEATURE_STORE_DIR = DATA_DIR / "feature_store"

ALPHA_THRESHOLD = 0.10

MAX_POSITION_PER_CONTRACT = 100
MAX_DAILY_EXPOSURE = 1000