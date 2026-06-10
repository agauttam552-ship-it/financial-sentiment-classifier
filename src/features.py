"""
Stage 2: Feature Engineering
- TF-IDF vectorization (sparse matrix, fast, interpretable)
- Fit ONLY on training data to prevent leakage
- Save/load vectorizer for reuse in API
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from src.utils import get_logger

logger = get_logger(__name__)

VECTORIZER_PATH = Path("models/tfidf_vectorizer.pkl")
LABEL_ENCODER_PATH = Path("models/label_encoder.pkl")


def build_features(
    df: pd.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
    test_size: float = 0.2,
    random_state: int = 42,
    max_features: int = 5000,
):
    """
    Split data, fit TF-IDF on training split only (no leakage),
    encode labels, and return all splits + fitted transformers.

    Why TF-IDF?
    - Fast and interpretable baseline
    - Works well on domain-specific vocabulary (SEC filings)
    - min_df=3 removes very rare terms (typos, company-specific codes)
    - max_df=0.85 removes near-universal terms that carry no signal
    - ngram_range=(1, 2) captures bigrams like "net loss", "revenue growth"
    """
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    logger.info(f"Total samples: {len(texts)}")
    logger.info(f"Label distribution: {pd.Series(labels).value_counts().to_dict()}")

    # ── Train/Test Split BEFORE fitting vectorizer (prevents data leakage) ────
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )
    logger.info(f"Train: {len(X_train_raw)}, Test: {len(X_test_raw)}")

    # ── Fit TF-IDF ONLY on training data ─────────────────────────────────────
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=3,          # word must appear in at least 3 docs
        max_df=0.85,       # word must not appear in >85% of docs
        ngram_range=(1, 2),# unigrams + bigrams
        sublinear_tf=True, # apply log(1+tf) to dampen high-freq terms
    )

    X_train = vectorizer.fit_transform(X_train_raw)  # fit + transform on train
    X_test = vectorizer.transform(X_test_raw)         # transform only on test

    logger.info(f"Feature matrix shape: train={X_train.shape}, test={X_test.shape}")

    # ── Label Encoding ────────────────────────────────────────────────────────
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    logger.info(f"Classes: {list(le.classes_)}")

    # ── Save transformers ─────────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(le, LABEL_ENCODER_PATH)
    logger.info(f"Saved vectorizer → {VECTORIZER_PATH}")
    logger.info(f"Saved label encoder → {LABEL_ENCODER_PATH}")

    return X_train, X_test, y_train, y_test, vectorizer, le


def load_vectorizer() -> TfidfVectorizer:
    return joblib.load(VECTORIZER_PATH)


def load_label_encoder() -> LabelEncoder:
    return joblib.load(LABEL_ENCODER_PATH)


def transform_text(text: str) -> np.ndarray:
    """Transform a single raw text string for inference."""
    vectorizer = load_vectorizer()
    return vectorizer.transform([text])


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("data/processed.csv")
    build_features(df)
