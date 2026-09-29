"""
Train and save the credit default risk model.

Steps:
    1. Load data/credit_default.csv
    2. Split into train/test, stratified on the target
    3. Fit a gradient boosting pipeline
    4. Evaluate: ROC AUC, PR AUC, and the confusion matrix at the review threshold
    5. Save the pipeline plus its metadata to models/credit_model.joblib

Usage:
    python scripts/train_model.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "credit_default.csv"
MODEL_PATH = BASE_DIR / "models" / "credit_model.joblib"

TARGET = "default_payment_next_month"
CATEGORICAL = ["SEX", "EDUCATION", "MARRIAGE"]
REVIEW_THRESHOLD = 0.30
RANDOM_STATE = 501


def main() -> None:
    print("=" * 62)
    print("Credit Default Risk — model training")
    print("=" * 62)

    # -------------------------------------------------------------------------
    print(f"\n[1/5] Loading {DATA_PATH.relative_to(BASE_DIR)} ...")
    if not DATA_PATH.exists():
        raise SystemExit(
            "Dataset not found. Run 'python scripts/make_dataset.py' first."
        )
    df = pd.read_csv(DATA_PATH)
    print(f"      {len(df):,} rows x {df.shape[1]} columns")
    print(f"      Default rate: {df[TARGET].mean():.1%}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric = [c for c in X.columns if c not in CATEGORICAL]

    # -------------------------------------------------------------------------
    print("\n[2/5] Splitting train / test (80 / 20, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"      train={len(X_train):,}  test={len(X_test):,}")

    # -------------------------------------------------------------------------
    print("\n[3/5] Fitting the pipeline ...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", "passthrough", numeric),
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.06,
                    max_depth=6,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    print("      done")

    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating on the held-out test set ...")
    proba = pipeline.predict_proba(X_test)[:, 1]
    predicted = (proba >= REVIEW_THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()

    print(f"      ROC AUC : {roc_auc:.4f}")
    print(f"      PR  AUC : {pr_auc:.4f}   (baseline = {y_test.mean():.4f})")
    print(f"\n      At threshold {REVIEW_THRESHOLD}:")
    print(f"        true negatives  {tn:>6,}   false positives {fp:>6,}")
    print(f"        false negatives {fn:>6,}   true positives  {tp:>6,}")
    print()
    print(classification_report(y_test, predicted, target_names=["no default", "default"]))

    # -------------------------------------------------------------------------
    print(f"[5/5] Saving to {MODEL_PATH.relative_to(BASE_DIR)} ...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(X.columns),
        "metrics": {
            "roc_auc": round(float(roc_auc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            "threshold": REVIEW_THRESHOLD,
        },
    }
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, MODEL_PATH)
    print(f"      saved ({MODEL_PATH.stat().st_size / 1024:.0f} KB)")

    print("\n" + "=" * 62)
    print("Next steps")
    print("=" * 62)
    print("  1. Start the API : uvicorn app.main:app --reload")
    print("  2. Open the docs : http://localhost:8000/docs")
    print("  3. Score someone : POST /predict")


if __name__ == "__main__":
    main()
