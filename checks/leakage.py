"""
Step 3: Leakage Detection

Idea: for each feature, try to predict the target using ONLY that one
feature. If a single feature alone predicts the target with suspiciously
high accuracy, it's a red flag — either the feature is a near-duplicate
of the target, or it was likely computed using information that wouldn't
actually be available at prediction time (a very common real-world mistake).

This is a heuristic, not a guarantee. A high score means "worth
investigating," not "definitely broken."
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


def _encode_single_feature(series: pd.Series) -> np.ndarray:
    """Turn one column into a 2D numeric array usable by a classifier."""
    s = series.copy()

    if s.dtype.kind in "biufc":
        s = s.fillna(s.median())
        return s.to_numpy().reshape(-1, 1)

    s = s.fillna("missing").astype(str)
    le = LabelEncoder()
    encoded = le.fit_transform(s)
    return encoded.reshape(-1, 1)


def detect_leakage(
    df: pd.DataFrame,
    target_col: str,
    suspicion_threshold: float = 0.90,
    n_splits: int = 5,
) -> dict:
    """
    Score every feature individually on how well it alone predicts the
    target. Returns features above the suspicion threshold, ranked by score.
    """
    if target_col not in df.columns:
        return {"supported": False, "reason": f"Target column '{target_col}' not found."}

    y_raw = df[target_col]
    n_unique = y_raw.nunique(dropna=True)

    if n_unique < 2 or n_unique > 20:
        return {
            "supported": False,
            "reason": (
                "Leakage detection currently supports classification targets "
                "(2 to 20 distinct classes). This target doesn't qualify."
            ),
        }

    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw.astype(str))

    min_class_count = pd.Series(y).value_counts().min()
    splits = max(2, min(n_splits, min_class_count))

    candidate_cols = [c for c in df.columns if c != target_col]

    # Skip obvious ID columns — every row unique, not a meaningful "feature"
    candidate_cols = [c for c in candidate_cols if df[c].nunique(dropna=False) < len(df)]

    scores = []
    for col in candidate_cols:
        try:
            X_single = _encode_single_feature(df[col])
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            cv_scores = cross_val_score(model, X_single, y, cv=splits, scoring="accuracy", n_jobs=-1)
            scores.append((col, float(cv_scores.mean())))
        except Exception:
            # If a feature can't be scored cleanly, skip it rather than crash the whole check
            continue

    scores_df = pd.DataFrame(scores, columns=["feature", "solo_accuracy"]).sort_values(
        "solo_accuracy", ascending=False
    )

    flagged = scores_df[scores_df["solo_accuracy"] >= suspicion_threshold]

    return {
        "supported": True,
        "all_scores": scores_df,
        "flagged_features": flagged,
        "n_flagged": len(flagged),
        "suspicion_threshold": suspicion_threshold,
    }


def summarize_leakage(result: dict) -> str:
    """One-line human-readable summary for the report."""
    if not result.get("supported"):
        return f"Leakage check skipped: {result.get('reason', 'not supported')}"

    if result["n_flagged"] == 0:
        return "No suspicious leakage detected — no single feature predicts the target too well on its own."

    top = result["flagged_features"].iloc[0]
    return (
        f"{result['n_flagged']} feature(s) may be leaking the target. "
        f"Most suspicious: '{top['feature']}' predicts the target alone with "
        f"{top['solo_accuracy']*100:.1f}% accuracy — worth investigating."
    )
