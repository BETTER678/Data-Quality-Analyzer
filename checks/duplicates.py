"""
Step 5: Duplicate Detection

Checks for:
- Exact duplicate rows within a dataset
- Duplicate overlap between training and test datasets
"""

import pandas as pd


def detect_duplicates(df: pd.DataFrame) -> dict:
    """
    Detect exact duplicate rows within a dataset.
    """

    total_rows = len(df)

    duplicate_mask = df.duplicated(keep=False)

    duplicate_rows = df[duplicate_mask]

    n_duplicate_rows = int(duplicate_mask.sum())

    duplicate_percentage = (
        round((n_duplicate_rows / total_rows) * 100, 2)
        if total_rows > 0
        else 0.0
    )

    return {
        "total_rows": total_rows,
        "n_duplicate_rows": n_duplicate_rows,
        "duplicate_percentage": duplicate_percentage,
        "duplicate_rows": duplicate_rows,
    }


def check_train_test_contamination(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    """
    Check whether exact rows appear in both
    training and test datasets.
    """

    common_columns = list(
        set(train_df.columns).intersection(test_df.columns)
    )

    if not common_columns:
        return {
            "supported": False,
            "reason": "Train and test datasets have no common columns."
        }

    train_common = train_df[common_columns].copy()
    test_common = test_df[common_columns].copy()

    train_tuples = set(
        train_common.astype(str).apply(tuple, axis=1)
    )

    test_tuples = test_common.astype(str).apply(tuple, axis=1)

    contamination_mask = test_tuples.isin(train_tuples)

    contaminated_rows = test_df[contamination_mask]

    n_contaminated = int(contamination_mask.sum())

    contamination_percentage = (
        round(
            (n_contaminated / len(test_df)) * 100,
            2
        )
        if len(test_df) > 0
        else 0.0
    )

    return {
        "supported": True,
        "n_contaminated_rows": n_contaminated,
        "contamination_percentage": contamination_percentage,
        "contaminated_rows": contaminated_rows,
    }


def summarize_duplicates(result: dict) -> str:
    """
    Create a human-readable duplicate summary.
    """

    if result["n_duplicate_rows"] == 0:
        return "No exact duplicate rows detected."

    return (
        f"{result['n_duplicate_rows']} rows are part of duplicate "
        f"records ({result['duplicate_percentage']}% of the dataset)."
    )


def summarize_contamination(result: dict) -> str:
    """
    Create a human-readable contamination summary.
    """

    if not result.get("supported"):
        return (
            f"Contamination check skipped: "
            f"{result.get('reason')}"
        )

    if result["n_contaminated_rows"] == 0:
        return (
            "No exact overlap detected between "
            "training and test datasets."
        )

    return (
        f"{result['n_contaminated_rows']} test rows also appear "
        f"in the training dataset "
        f"({result['contamination_percentage']}%)."
    )