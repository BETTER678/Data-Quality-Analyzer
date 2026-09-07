"""
Step 2: Label Noise Detection

Idea: train a model using cross-validation. For every row, get the
prediction made when that row was held out of training (so the model
never "cheated" by seeing it). If the model is very confident about a
DIFFERENT answer than the row's actual label, that row is likely
mislabeled rather than just a hard example.

This only works for classification targets (binary or multi-class).
It needs at least a small amount of clean signal in the data to work —
if the whole dataset is random noise, this won't find anything meaningful.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder


def _prepare_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Very simple feature prep: drop the target, drop obvious ID-like columns,
    one-hot encode categoricals, fill missing values.
    This is intentionally basic — the goal is a usable signal for noise
    detection, not a production-grade feature pipeline.
    """
    X = df.drop(columns=[target_col]).copy()

    # Drop columns that look like unique identifiers (little predictive value,
    # and can make cross-validation behave oddly)
    for col in X.columns:
        if X[col].nunique(dropna=False) == len(X):
            X = X.drop(columns=[col])

    # Fill missing values simply: numeric -> median, categorical -> mode
    for col in X.columns:
        if X[col].dtype.kind in "biufc":
            X[col] = X[col].fillna(X[col].median())
        else:
            mode = X[col].mode(dropna=True)
            X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "missing")

    # One-hot encode remaining categoricals
    X = pd.get_dummies(X, drop_first=True)

    return X


def detect_label_noise(
    df: pd.DataFrame,
    target_col: str,
    n_splits: int = 5,
    confidence_threshold: float = 0.9,
    top_n: int = 20,
) -> dict:
    """
    Run cross-validated predictions and flag rows where the model is
    confidently wrong about the label.

    Returns a dict with:
      - "supported": whether this check could run (needs a classification target)
      - "flagged_rows": DataFrame of the most suspicious rows
      - "n_flagged": how many rows were flagged in total
      - "flagged_pct": percentage of the dataset flagged
    """
    if target_col not in df.columns:
        return {"supported": False, "reason": f"Target column '{target_col}' not found."}

    y_raw = df[target_col]
    n_unique = y_raw.nunique(dropna=True)

    # Only makes sense for classification with a reasonably small number of classes
    if n_unique < 2 or n_unique > 20:
        return {
            "supported": False,
            "reason": (
                "Label noise detection needs a classification target "
                "(2 to 20 distinct classes). This target doesn't qualify."
            ),
        }

    X = _prepare_features(df, target_col)
    if X.shape[1] == 0:
        return {"supported": False, "reason": "No usable features remained after preprocessing."}

    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    # Guard against tiny datasets where cross-validation isn't meaningful
    min_class_count = pd.Series(y).value_counts().min()
    n_splits = max(2, min(n_splits, min_class_count))

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    try:
        probs = cross_val_predict(
            model, X, y, cv=n_splits, method="predict_proba", n_jobs=-1
        )
    except ValueError as e:
        return {"supported": False, "reason": f"Could not run cross-validation: {e}"}

    predicted_class = probs.argmax(axis=1)
    predicted_confidence = probs.max(axis=1)
    actual_class = y

    is_wrong = predicted_class != actual_class
    is_confident = predicted_confidence >= confidence_threshold
    suspicious = is_wrong & is_confident

    result_df = df.copy()
    result_df["_predicted_label"] = le.inverse_transform(predicted_class)
    result_df["_model_confidence"] = predicted_confidence.round(3)
    result_df["_suspicious"] = suspicious

    flagged = (
        result_df[result_df["_suspicious"]]
        .sort_values("_model_confidence", ascending=False)
        .head(top_n)
    )

    n_flagged = int(suspicious.sum())
    flagged_pct = round(n_flagged / len(df) * 100, 2) if len(df) else 0.0

    return {
        "supported": True,
        "flagged_rows": flagged.drop(columns=["_suspicious"]),
        "n_flagged": n_flagged,
        "flagged_pct": flagged_pct,
        "n_splits_used": n_splits,
        "confidence_threshold": confidence_threshold,
    }


def summarize_label_noise(result: dict) -> str:
    """One-line human-readable summary for the report."""
    if not result.get("supported"):
        return f"Label noise check skipped: {result.get('reason', 'not supported')}"

    if result["n_flagged"] == 0:
        return "No suspicious label noise detected."

    return (
        f"{result['n_flagged']} row(s) ({result['flagged_pct']}%) look like they may have "
        f"the wrong label — the model was at least {int(result['confidence_threshold']*100)}% "
        f"confident in a different answer."
    )
