from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # src/research/paths.py -> รากรีโป
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
DATA_CACHE_DIR = EXPERIMENTS_DIR / "data_cache"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
