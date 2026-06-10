"""
Stage 4: Model Evaluation
- Evaluates all three models: Accuracy, Precision, Recall, F1
- Generates confusion matrix for each
- Prints comparison table
- Saves plots to reports/
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.sparse import issparse

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils import get_logger

logger = get_logger(__name__)
Path("reports").mkdir(exist_ok=True)


def evaluate_single_model(
    name: str,
    model,
    X_test,
    y_test,
    class_names: list,
) -> dict:
    """
    Run full evaluation on a single model.
    Returns dict of metrics, also prints classification report and
    saves confusion matrix plot.
    """
    # CatBoost needs dense input
    if issparse(X_test):
        X_test_dense = X_test.toarray()
    else:
        X_test_dense = X_test

    if name == "CatBoost":
        y_pred = model.predict(X_test_dense)
    else:
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    logger.info(f"\n{'='*50}")
    logger.info(f"  {name} Results")
    logger.info(f"{'='*50}")
    print(classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0
    ))

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"{name} — Confusion Matrix", fontsize=14, pad=12)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    plt.tight_layout()
    plot_path = f"reports/cm_{name.lower()}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {plot_path}")

    return {
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
    }


def evaluate_all_models(
    trained_models: dict,
    X_test,
    y_test,
    class_names: list,
) -> pd.DataFrame:
    """
    Evaluate all three models and print a comparison table.
    Returns a DataFrame with all metrics.
    """
    results = []
    for name, model in trained_models.items():
        metrics = evaluate_single_model(name, model, X_test, y_test, class_names)
        results.append(metrics)

    comparison_df = pd.DataFrame(results).set_index("Model")

    logger.info("\n" + "="*60)
    logger.info("  MODEL COMPARISON")
    logger.info("="*60)
    print(comparison_df.to_string())

    # ── Save comparison table ─────────────────────────────────────────────────
    comparison_df.to_csv("reports/model_comparison.csv")
    logger.info("Comparison saved → reports/model_comparison.csv")

    # ── Bar chart comparison ──────────────────────────────────────────────────
    ax = comparison_df.plot(kind="bar", figsize=(10, 5), rot=0)
    ax.set_title("Model Comparison — All Metrics", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("reports/model_comparison.png", dpi=150)
    plt.close()
    logger.info("Comparison chart saved → reports/model_comparison.png")

    # ── Select best model by F1 ───────────────────────────────────────────────
    best_name = comparison_df["F1-Score"].idxmax()
    logger.info(f"\n★ Best model: {best_name} "
                f"(F1={comparison_df.loc[best_name, 'F1-Score']})")

    return comparison_df, best_name


if __name__ == "__main__":
    import pandas as pd
    from src.features import build_features
    from src.train import train_all_models, save_models

    df = pd.read_csv("data/processed.csv")
    X_train, X_test, y_train, y_test, vectorizer, le = build_features(df)
    trained_models = train_all_models(X_train, y_train)

    class_names = list(le.classes_)
    comparison_df, best_name = evaluate_all_models(
        trained_models, X_test, y_test, class_names
    )

    save_models(trained_models, best_name=best_name)
    logger.info("Evaluation complete.")
