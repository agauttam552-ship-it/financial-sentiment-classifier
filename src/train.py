"""
Stage 3: Classification Model Training
- Trains XGBoost, AdaBoost, and CatBoost on TF-IDF features
- Saves the best-performing model
- Logs training time for each model
"""

import time
import joblib
import numpy as np
from pathlib import Path
from scipy.sparse import issparse

from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier

from src.utils import get_logger

logger = get_logger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)


def get_model_configs() -> dict:
    """
    Return all three required models with their configurations.
    Parameters chosen for balance between speed and accuracy on ~800 samples.
    """
    return {
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=42,
            verbosity=0,
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.5,
            random_state=42,
            algorithm="SAMME",  # required for multi-class
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.1,
            depth=6,
            random_state=42,
            verbose=0,
        ),
    }


def train_all_models(
    X_train,
    y_train,
) -> dict:
    """
    Train XGBoost, AdaBoost, and CatBoost.
    Returns a dict of {model_name: fitted_model}.
    """
    # CatBoost needs dense input
    if issparse(X_train):
        X_train_dense = X_train.toarray()
    else:
        X_train_dense = X_train

    models = get_model_configs()
    trained = {}

    for name, model in models.items():
        logger.info(f"Training {name} ...")
        t0 = time.time()

        if name == "CatBoost":
            model.fit(X_train_dense, y_train)
        else:
            model.fit(X_train, y_train)

        elapsed = round(time.time() - t0, 2)
        logger.info(f"  {name} trained in {elapsed}s")
        trained[name] = model

    return trained


def save_models(trained_models: dict, best_name: str):
    """Save all models; mark the best one as best_model.pkl."""
    for name, model in trained_models.items():
        path = MODEL_DIR / f"{name.lower()}_model.pkl"
        joblib.dump(model, path)
        logger.info(f"Saved {name} → {path}")

    best_model = trained_models[best_name]
    joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
    logger.info(f"Best model ({best_name}) saved → models/best_model.pkl")


def load_best_model():
    return joblib.load(MODEL_DIR / "best_model.pkl")




if __name__ == "__main__":
    import pandas as pd
    from src.features import build_features

    df = pd.read_csv("data/processed.csv")
    X_train, X_test, y_train, y_test, vectorizer, le = build_features(df)

    trained_models = train_all_models(X_train, y_train)
    save_models(trained_models, best_name="XGBoost")

    logger.info("Training complete. Run evaluate.py for metrics.")

