"""
Stage 1: Data Extraction & Preprocessing
- Loads 10-K SEC filings from HuggingFace
- Cleans text (lowercase, remove noise, normalize whitespace)
- Extracts key sections: Business, MDA, Financials
- Assigns sentiment labels using Loughran-McDonald word lists
"""

import re
import pandas as pd
from datasets import load_dataset
from collections import Counter
from src.utils import get_logger

logger = get_logger(__name__)

# ── Loughran-McDonald Financial Sentiment Word Lists ──────────────────────────
# Source: Loughran & McDonald (2011) - widely used in financial NLP
POSITIVE_WORDS = {
    "growth", "profitable", "strong", "excellent", "improved", "exceeded",
    "record", "increased", "gain", "gains", "positive", "success", "successful",
    "opportunity", "opportunities", "innovative", "expand", "expanding",
    "surpassed", "outperformed", "achieved", "achievement", "robust",
    "favorable", "efficient", "efficiency", "advancing", "advance",
    "benefited", "benefit", "benefits", "upward", "momentum", "confident",
    "confidence", "optimistic", "growth", "profitable", "revenue",
    "earnings", "profitability", "solid", "sustained", "recovery",
    "improving", "better", "best", "highest", "exceeded", "exceeding",
}

NEGATIVE_WORDS = {
    "risk", "risks", "uncertain", "uncertainty", "decline", "declined",
    "loss", "losses", "deficit", "impairment", "adverse", "adversely",
    "deterioration", "downturn", "negative", "challenging", "challenges",
    "difficulty", "difficulties", "volatile", "volatility", "concern",
    "concerns", "weakening", "weak", "lawsuit", "litigation", "penalty",
    "penalties", "investigation", "default", "bankruptcy", "insolvency",
    "restructuring", "layoffs", "impaired", "reduced", "reduction",
    "decrease", "decreased", "lower", "lowest", "deteriorated",
    "unfavorable", "failure", "failed", "failed", "shortfall",
    "writedown", "writeoff", "liquidation", "discontinued",
}

# Section column names as they appear in the dataset
SECTION_COLS = {
    "Business": "Business",
    "MDA": "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    "Financials": "Financial Statements and Supplementary Data",
    "RiskFactors": "Risk Factors",
}


def load_raw_data(split: str = "001", max_rows: int = 1000) -> pd.DataFrame:
    """
    Load 10-K SEC filings from HuggingFace in streaming mode.
    Uses split '001' (the available data split for this dataset).
    """
    logger.info(f"Loading dataset split='{split}', max_rows={max_rows} ...")
    dataset = load_dataset(
        "winterForestStump/10-K_sec_filings",
        split=split,
        streaming=True,
    )

    rows = []
    for i, row in enumerate(dataset):
        rows.append(row)
        if i == max_rows - 1:
            break

    df = pd.DataFrame(rows)
    df.fillna("", inplace=True)
    logger.info(f"Loaded {len(df)} rows. Columns: {list(df.columns)}")
    return df


def clean_text(text: str) -> str:
    """
    Clean raw SEC filing text:
    - Lowercase
    - Remove possessives ('s)
    - Remove digits (financial NLP: numbers are noisy for section classification)
    - Remove all non-alpha characters
    - Normalize whitespace
    """
    text = str(text).lower()
    text = re.sub(r"'s\b", "", text)           # remove possessives
    text = re.sub(r"\d+", " ", text)           # remove numbers
    text = re.sub(r"[^a-z\s]", " ", text)      # keep only letters + spaces
    text = re.sub(r"\s+", " ", text)           # collapse whitespace
    return text.strip()


def score_sentiment(text: str) -> str:
    """
    Assign a sentiment label to a document using Loughran-McDonald word lists.
    Counts positive vs negative financial words and labels based on ratio.

    Returns: 'positive', 'negative', or 'neutral'
    """
    words = set(text.lower().split())
    pos_count = len(words & POSITIVE_WORDS)
    neg_count = len(words & NEGATIVE_WORDS)
    total = pos_count + neg_count

    if total == 0:
        return "neutral"

    ratio = pos_count / total
    if ratio >= 0.55:
        return "positive"
    elif ratio <= 0.45:
        return "negative"
    else:
        return "neutral"


def build_section_df(df: pd.DataFrame, min_length: int = 200) -> pd.DataFrame:
    """
    Build the classification dataset from raw filings.

    Strategy:
    - Combine Business + MDA sections (most informative for sentiment)
    - Clean the combined text
    - Assign a sentiment label using Loughran-McDonald scoring
    - Filter out rows with insufficient text

    Why combined_text instead of individual sections?
    - Business section describes what the company does
    - MDA section reflects management's forward-looking tone
    - Together they give the richest signal for sentiment classification
    """
    records = []

    for _, row in df.iterrows():
        business = str(row.get(SECTION_COLS["Business"], ""))
        mda = str(row.get(SECTION_COLS["MDA"], ""))
        financials = str(row.get(SECTION_COLS["Financials"], ""))
        company = str(row.get("company_name", ""))

        # Combine the two most sentiment-rich sections
        combined_raw = business + " " + mda

        if len(combined_raw.strip()) < min_length:
            continue

        cleaned = clean_text(combined_raw)
        # Truncate to 3000 tokens to keep memory manageable
        cleaned = " ".join(cleaned.split()[:3000])

        if len(cleaned) < min_length:
            continue

        label = score_sentiment(cleaned)

        records.append({
            "company_name": company,
            "text": cleaned,
            "label": label,
            "business_raw": business[:500],   # keep snippet for inspection
            "mda_raw": mda[:500],
        })

    result = pd.DataFrame(records)
    logger.info(f"Built dataset: {len(result)} samples")
    logger.info(f"Label distribution: {Counter(result['label'])}")
    return result


def preprocess_pipeline(split: str = "001", max_rows: int = 1000) -> pd.DataFrame:
    """Full pipeline: load → extract sections → clean → label → return DataFrame."""
    df_raw = load_raw_data(split=split, max_rows=max_rows)
    df_clean = build_section_df(df_raw)
    return df_clean


if __name__ == "__main__":
    df = preprocess_pipeline()
    df.to_csv("data/processed.csv", index=False)
    logger.info("Saved to data/processed.csv")
    print(df[["company_name", "label", "text"]].head(10).to_string())
