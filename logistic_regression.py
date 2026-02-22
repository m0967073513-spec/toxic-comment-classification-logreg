"""
Toxic Comment Classification using Logistic Regression
=======================================================
Trains a binary text classifier that labels comments as toxic or non-toxic.
Pipeline: raw text → TF-IDF features → Logistic Regression.

Usage
-----
    python logistic_regression.py                  # run with built-in sample data
    python logistic_regression.py --data path/to/train.csv  # run with custom CSV

Expected CSV columns: ``text`` (str), ``toxic`` (0 or 1).
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Sample data (used when no external CSV is provided)
# ---------------------------------------------------------------------------
SAMPLE_COMMENTS = [
    ("I hate you so much, you are worthless!", 1),
    ("You are the worst person I have ever met.", 1),
    ("Go kill yourself, nobody cares about you.", 1),
    ("You stupid idiot, get out of here!", 1),
    ("What a disgusting and pathetic excuse for a human being.", 1),
    ("This is absolutely horrible behavior, shame on you.", 1),
    ("You are a complete moron and everyone knows it.", 1),
    ("I despise everything about you.", 1),
    ("This video is great, thanks for sharing!", 0),
    ("I really enjoyed reading this article.", 0),
    ("What a beautiful day to learn something new.", 0),
    ("Thank you for your helpful explanation.", 0),
    ("I disagree with this point, but I respect your view.", 0),
    ("Could you please clarify what you meant by that?", 0),
    ("Great work on this project, keep it up!", 0),
    ("This is a well-written and thoughtful post.", 0),
    ("I found this tutorial very useful, appreciate it.", 0),
    ("Looking forward to the next episode!", 0),
    ("Nice analysis, I learned a lot from this.", 0),
    ("The community here is always so supportive.", 0),
]


def load_sample_data() -> pd.DataFrame:
    """Return a small, self-contained demo dataset."""
    texts, labels = zip(*SAMPLE_COMMENTS)
    return pd.DataFrame({"text": list(texts), "toxic": list(labels)})


def load_csv_data(path: str) -> pd.DataFrame:
    """Load and validate a CSV file with ``text`` and ``toxic`` columns."""
    df = pd.read_csv(path)
    required = {"text", "toxic"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            "Expected columns: 'text' (str) and 'toxic' (0 or 1)."
        )
    df = df[["text", "toxic"]].dropna()
    df["toxic"] = df["toxic"].astype(int)
    return df


def build_pipeline() -> Pipeline:
    """Return a scikit-learn Pipeline: TF-IDF → Logistic Regression."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    analyzer="word",
                    token_pattern=r"\w{1,}",
                    ngram_range=(1, 2),
                    max_features=10_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=1_000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, X_test: pd.Series, y_test: pd.Series) -> None:
    """Print accuracy, ROC-AUC, and a full classification report."""
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\n" + "=" * 50)
    print("Model Evaluation")
    print("=" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["non-toxic", "toxic"]))


def predict_examples(pipeline: Pipeline, comments: list[str]) -> None:
    """Run inference on a list of example comments and print results."""
    print("=" * 50)
    print("Example Predictions")
    print("=" * 50)
    probs = pipeline.predict_proba(comments)[:, 1]
    for comment, prob in zip(comments, probs):
        label = "TOXIC" if prob >= 0.5 else "SAFE"
        print(f"[{label} | {prob:.2f}] {comment!r}")


def main(data_path: str | None = None) -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    if data_path:
        print(f"Loading data from: {data_path}")
        df = load_csv_data(data_path)
    else:
        print("No data file provided — using built-in sample data.")
        df = load_sample_data()

    print(f"Dataset size: {len(df)} rows | Toxic: {df['toxic'].sum()} | Non-toxic: {(df['toxic'] == 0).sum()}")

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["toxic"],
        test_size=0.2,
        random_state=42,
        stratify=df["toxic"],
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # ------------------------------------------------------------------
    # 3. Train pipeline
    # ------------------------------------------------------------------
    print("\nTraining pipeline (TF-IDF + Logistic Regression) …")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # ------------------------------------------------------------------
    # 4. Evaluate
    # ------------------------------------------------------------------
    evaluate(pipeline, X_test, y_test)

    # ------------------------------------------------------------------
    # 5. Quick demo predictions
    # ------------------------------------------------------------------
    demo_comments = [
        "You are the most pathetic person alive.",
        "I really appreciate your kind words, thank you!",
        "This post is garbage and so are you.",
        "Wonderful tutorial, learned a lot!",
    ]
    predict_examples(pipeline, demo_comments)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Toxic comment classifier — TF-IDF + Logistic Regression"
    )
    parser.add_argument(
        "--data",
        metavar="PATH",
        default=None,
        help="Path to a CSV file with 'text' and 'toxic' columns (optional).",
    )
    args = parser.parse_args()
    main(data_path=args.data)
