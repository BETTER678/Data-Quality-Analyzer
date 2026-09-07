import pandas as pd


def detect_class_imbalance(
    df: pd.DataFrame,
    target_col: str,
    imbalance_threshold: float = 0.20,
) -> dict:
    """
    Checks whether the target classes are imbalanced.

    A class is considered underrepresented if its percentage
    is below the imbalance threshold.
    """

    if target_col not in df.columns:
        return {
            "supported": False,
            "reason": f"Target column '{target_col}' not found."
        }

    class_counts = df[target_col].value_counts(dropna=False)

    if len(class_counts) < 2:
        return {
            "supported": False,
            "reason": "Class imbalance detection requires at least 2 classes."
        }

    class_percentages = (
        df[target_col]
        .value_counts(normalize=True, dropna=False)
        * 100
    ).round(2)

    distribution = pd.DataFrame({
        "count": class_counts,
        "percentage": class_percentages
    })

    minority_percentage = class_percentages.min()

    is_imbalanced = bool(minority_percentage < (imbalance_threshold * 100))

    imbalance_ratio = round(
        class_counts.min() / class_counts.max(),
        3
    )

    return {
        "supported": True,
        "distribution": distribution,
        "n_classes": len(class_counts),
        "minority_percentage": float(minority_percentage),
        "imbalance_ratio": imbalance_ratio,
        "is_imbalanced": is_imbalanced,
        "imbalance_threshold": imbalance_threshold,
    }


def summarize_class_imbalance(result: dict) -> str:

    if not result.get("supported"):
        return (
            f"Class imbalance check skipped: "
            f"{result.get('reason', 'not supported')}"
        )

    if result["is_imbalanced"]:
        return (
            f"Class imbalance detected. "
            f"The smallest class represents only "
            f"{result['minority_percentage']}% of the dataset."
        )

    return (
        f"No significant class imbalance detected. "
        f"The smallest class represents "
        f"{result['minority_percentage']}% of the dataset."
    )