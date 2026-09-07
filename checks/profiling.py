"""
Step 1: Basic Data Profiling

Looks at a dataset and reports the fundamentals:
- missing values per column
- data types
- duplicate rows
- constant columns (no useful information)
- cardinality of categorical columns

This is intentionally simple. Later checks (label noise, leakage, etc.)
build on top of what this step finds.
"""

import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Run basic profiling on a dataframe and return a dictionary of findings.
    """
    n_rows, n_cols = df.shape

    # --- Missing values ---
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / n_rows * 100).round(2)
    missing_report = (
        pd.DataFrame({
            "missing_count": missing_counts,
            "missing_pct": missing_pct,
        })
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
    )

    # --- Data types ---
    dtypes_report = df.dtypes.astype(str)

    # --- Duplicate rows ---
    n_duplicates = int(df.duplicated().sum())
    duplicate_pct = round(n_duplicates / n_rows * 100, 2) if n_rows else 0.0

    # --- Constant columns (zero variance / only one unique value) ---
    constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]

    # --- Cardinality of categorical / object columns ---
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    cardinality = {col: int(df[col].nunique(dropna=True)) for col in cat_cols}

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_report": missing_report,
        "dtypes_report": dtypes_report,
        "n_duplicates": n_duplicates,
        "duplicate_pct": duplicate_pct,
        "constant_cols": constant_cols,
        "cardinality": cardinality,
    }


def summarize_profile(profile: dict) -> list:
    """
    Turn the raw profile dict into a short list of human-readable findings.
    Used for the final report and the dashboard.
    """
    findings = []

    if not profile["missing_report"].empty:
        worst_col = profile["missing_report"].index[0]
        worst_pct = profile["missing_report"].iloc[0]["missing_pct"]
        findings.append(
            f"Column '{worst_col}' has {worst_pct}% missing values "
            f"({len(profile['missing_report'])} columns have some missing data in total)."
        )
    else:
        findings.append("No missing values found.")

    if profile["duplicate_pct"] > 0:
        findings.append(
            f"{profile['n_duplicates']} duplicate rows found "
            f"({profile['duplicate_pct']}% of the dataset)."
        )
    else:
        findings.append("No duplicate rows found.")

    if profile["constant_cols"]:
        findings.append(
            f"{len(profile['constant_cols'])} constant column(s) found "
            f"(no useful information): {', '.join(profile['constant_cols'])}."
        )

    return findings
