"""
Stage 5: API Deployment
FastAPI prediction endpoint for the best trained model.

Run:  uvicorn api.app:app --reload
Docs: http://localhost:8000/docs
"""

import re
import sys
import numpy as np
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ── Allow imports from project root ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = ROOT / "models" / "best_model.pkl"
VECTORIZER_PATH = ROOT / "models" / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = ROOT / "models" / "label_encoder.pkl"

# ── Load artifacts at startup ─────────────────────────────────────────────────
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)
    CLASS_NAMES = list(le.classes_)
    MODEL_LOADED = True
except FileNotFoundError as e:
    print(f"[WARNING] Model files not found: {e}")
    print("Run 'python run_pipeline.py' first to train and save models.")
    MODEL_LOADED = False

# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SEC 10-K Financial Sentiment Classifier",
    description=(
        "Classifies the sentiment of SEC 10-K filing text "
        "as **positive**, **negative**, or **neutral** "
        "using a TF-IDF + XGBoost pipeline."
    ),
    version="1.0.0",
)


# ── Text Cleaning (must match preprocess.py) ──────────────────────────────────
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"'s\b", "", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ── Request / Response Schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=20,
        description="Raw text from a 10-K SEC filing section (Business, MDA, etc.)",
        example=(
            "The company achieved record revenue growth driven by strong "
            "demand across all business segments. Management is optimistic "
            "about continued expansion in emerging markets."
        ),
    )


class PredictResponse(BaseModel):
    label: str = Field(..., description="Predicted sentiment: positive / negative / neutral")
    confidence: float = Field(..., description="Model confidence for the predicted label (0–1)")
    all_probabilities: dict = Field(..., description="Probabilities for each class")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/", response_model=HealthResponse, summary="Health check")
def root():
    """Check if the API is running and models are loaded."""
    return HealthResponse(
        status="ok" if MODEL_LOADED else "model_not_loaded",
        model_loaded=MODEL_LOADED,
        classes=CLASS_NAMES if MODEL_LOADED else [],
    )


@app.post("/predict", response_model=PredictResponse, summary="Classify 10-K text sentiment")
def predict(request: PredictRequest):
    """
    **Input:** Raw text from any section of a 10-K SEC filing.

    **Output:** Sentiment label (positive / negative / neutral) + confidence score.

    The text is cleaned, vectorized using TF-IDF, and classified with XGBoost.
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run 'python run_pipeline.py' first.",
        )

    # Clean + vectorize
    cleaned = clean_text(request.text)
    if len(cleaned.strip()) < 10:
        raise HTTPException(
            status_code=422,
            detail="Text too short after cleaning. Please provide more content.",
        )

    # Truncate to 3000 words (same as training)
    cleaned = " ".join(cleaned.split()[:3000])
    vector = vectorizer.transform([cleaned])

    # Handle CatBoost (needs dense)
    model_type = type(model).__name__
    if model_type == "CatBoostClassifier":
        vector = vector.toarray()

    # Predict
    pred_enc = model.predict(vector)[0]
    proba = model.predict_proba(vector)[0]

    label = le.inverse_transform([int(pred_enc)])[0]
    confidence = float(np.max(proba))

    all_probs = {
        cls: round(float(p), 4)
        for cls, p in zip(CLASS_NAMES, proba)
    }

    return PredictResponse(
        label=label,
        confidence=round(confidence, 4),
        all_probabilities=all_probs,
    )


@app.get("/classes", summary="List available output classes")
def get_classes():
    """Returns the list of classes the model can predict."""
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"classes": CLASS_NAMES}
