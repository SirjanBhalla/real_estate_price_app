import pathlib

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent
DATA_PATH = PACKAGE_ROOT.parent.parent / "data" / "final.csv"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
MODEL_NAME = "real_estate_pipeline.joblib"
MODEL_SAVE_PATH = TRAINED_MODEL_DIR / MODEL_NAME
