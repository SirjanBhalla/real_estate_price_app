import os
import joblib
from real_estate_price import config
from real_estate_price.data_management import load_and_prepare_data
from real_estate_price.pipeline import real_estate_pipeline

def run_training():
    print("Starting training on real estate data...")
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    real_estate_pipeline.fit(X_train, y_train)
    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)
    joblib.dump(real_estate_pipeline, config.MODEL_SAVE_PATH)
    print("Model trained and saved successfully.")

if __name__ == '__main__':
    run_training()
