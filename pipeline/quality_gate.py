"""
Quality Gate

Decides whether a dataset is allowed to move
forward for machine learning model training.
"""


def apply_quality_gate(readiness_result: dict) -> dict:
    """
    Apply quality gate rules based on
    the Dataset Readiness Score.
    """

    score = readiness_result["score"]

    if score >= 80:

        decision = "APPROVED"

        message = (
            "Dataset passed the quality gate and "
            "is approved for ML model training."
        )

    elif score >= 60:

        decision = "REVIEW REQUIRED"

        message = (
            "Dataset has moderate quality issues. "
            "Review the issues before ML model training."
        )

    else:

        decision = "BLOCKED"

        message = (
            "Dataset failed the quality gate and "
            "should not be used for ML model training."
        )

    return {
        "score": score,
        "decision": decision,
        "message": message,
        "issues": readiness_result.get("issues", [])
    }