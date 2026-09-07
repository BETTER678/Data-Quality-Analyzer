"""
Dataset Readiness Score

Combines results from multiple data quality checks
into a single score from 0 to 100.
"""


def calculate_readiness_score(
    profile,
    label_noise_result=None,
    leakage_result=None,
    imbalance_result=None,
    duplicate_result=None,
    outlier_result=None,
    correlation_result=None,
    type_issues_result=None,
):
    """
    Calculate a dataset readiness score.

    Score starts at 100.

    Points are deducted based on detected
    data quality issues.
    """

    score = 100
    issues = []

    # ==========================================
    # MISSING VALUES
    # ==========================================

    if not profile["missing_report"].empty:

        max_missing = profile[
            "missing_report"
        ]["missing_pct"].max()

        if max_missing >= 50:

            score -= 20

            issues.append(
                "Very high percentage of missing values."
            )

        elif max_missing >= 20:

            score -= 10

            issues.append(
                "Moderate amount of missing values."
            )

        elif max_missing > 0:

            score -= 5

            issues.append(
                "Some missing values detected."
            )


    # ==========================================
    # DUPLICATES
    # ==========================================

    if duplicate_result is not None:

        duplicate_pct = duplicate_result[
            "duplicate_percentage"
        ]

        if duplicate_pct >= 20:

            score -= 15

            issues.append(
                "High number of duplicate records."
            )

        elif duplicate_pct >= 5:

            score -= 8

            issues.append(
                "Moderate number of duplicate records."
            )

        elif duplicate_pct > 0:

            score -= 3

            issues.append(
                "Some duplicate records detected."
            )


    # ==========================================
    # CONSTANT COLUMNS
    # ==========================================

    constant_columns = len(
        profile["constant_cols"]
    )

    if constant_columns > 0:

        penalty = min(
            constant_columns * 3,
            10
        )

        score -= penalty

        issues.append(
            f"{constant_columns} constant column(s) detected."
        )


    # ==========================================
    # LABEL NOISE
    # ==========================================

    if (
        label_noise_result is not None
        and label_noise_result.get("supported")
    ):

        noise_pct = label_noise_result[
            "flagged_pct"
        ]

        if noise_pct >= 20:

            score -= 20

            issues.append(
                "High potential label noise detected."
            )

        elif noise_pct >= 10:

            score -= 10

            issues.append(
                "Moderate potential label noise detected."
            )

        elif noise_pct > 0:

            score -= 5

            issues.append(
                "Some potentially mislabeled records detected."
            )


    # ==========================================
    # DATA LEAKAGE
    # ==========================================

    if (
        leakage_result is not None
        and leakage_result.get("supported")
    ):

        if leakage_result["n_flagged"] > 0:

            score -= 25

            issues.append(
                "Potential data leakage detected."
            )


    # ==========================================
    # CLASS IMBALANCE
    # ==========================================

    if (
        imbalance_result is not None
        and imbalance_result.get("supported")
    ):

        if imbalance_result["is_imbalanced"]:

            score -= 10

            issues.append(
                "Class imbalance detected."
            )


    # ==========================================
    # OUTLIERS
    # ==========================================

    if outlier_result is not None and outlier_result.get("supported", True):

        outlier_pct = outlier_result.get(
            "total_outlier_pct",
            outlier_result.get("outlier_row_percentage", 0)
        )

        if outlier_pct >= 15:

            score -= 15

            issues.append(
                "High percentage of outlier rows."
            )

        elif outlier_pct >= 5:

            score -= 8

            issues.append(
                "Moderate percentage of outlier rows."
            )

        elif outlier_pct > 0:

            score -= 3

            issues.append(
                "Some outlier rows detected."
            )


    # ==========================================
    # HIGH CORRELATION
    # ==========================================

    if correlation_result is not None and correlation_result.get("supported", True):

        high_corr_pairs = correlation_result.get(
            "high_correlation_pairs",
            correlation_result.get("highly_correlated_pairs", [])
        )
        n_pairs = correlation_result.get("n_high_pairs", len(high_corr_pairs))

        if n_pairs >= 3:

            score -= 10

            issues.append(
                "Multiple highly correlated feature pairs detected."
            )

        elif n_pairs >= 1:

            score -= 5

            issues.append(
                "Some highly correlated feature pairs detected."
            )


    # ==========================================
    # DATA TYPE ISSUES
    # ==========================================

    if type_issues_result is not None and type_issues_result.get("supported", True):

        n_issues = type_issues_result.get(
            "n_issues",
            type_issues_result.get("total_issues", 0)
        )

        if n_issues >= 3:

            score -= 10

            issues.append(
                "Multiple data type issues detected."
            )

        elif n_issues >= 1:

            score -= 5

            issues.append(
                "Some data type issues detected."
            )


    # ==========================================
    # FINAL SCORE
    # ==========================================

    score = max(0, score)


    # ==========================================
    # DATASET STATUS
    # ==========================================

    if score >= 80:

        status = "PASS"

    elif score >= 60:

        status = "WARNING"

    else:

        status = "FAIL"


    # Calculate category scores
    completeness = 100
    if not profile["missing_report"].empty:
        max_missing = profile["missing_report"]["missing_pct"].max()
        if max_missing >= 50:
            completeness = 40
        elif max_missing >= 20:
            completeness = 70
        elif max_missing > 0:
            completeness = 90

    consistency = 100
    if type_issues_result is not None and type_issues_result.get("supported", True):
        n_issues = type_issues_result.get(
            "n_issues",
            type_issues_result.get("total_issues", 0)
        )
        if n_issues >= 3: consistency -= 25
        elif n_issues >= 1: consistency -= 10
    if duplicate_result is not None:
        dup_pct = duplicate_result.get("duplicate_percentage", 0)
        if dup_pct >= 20: consistency -= 25
        elif dup_pct >= 5: consistency -= 10
        elif dup_pct > 0: consistency -= 5

    accuracy = 100
    if label_noise_result is not None and label_noise_result.get("supported"):
        noise_pct = label_noise_result.get("flagged_pct", 0)
        if noise_pct >= 20: accuracy -= 30
        elif noise_pct >= 10: accuracy -= 15
        elif noise_pct > 0: accuracy -= 5
    if leakage_result is not None and leakage_result.get("supported"):
        if leakage_result.get("n_flagged", 0) > 0: accuracy -= 40

    uniqueness = 100
    if duplicate_result is not None:
        dup_pct = duplicate_result.get("duplicate_percentage", 0)
        if dup_pct >= 20: uniqueness -= 30
        elif dup_pct >= 5: uniqueness -= 15
        elif dup_pct > 0: uniqueness -= 5
    if correlation_result is not None and correlation_result.get("supported", True):
        high_corr_pairs = correlation_result.get(
            "high_correlation_pairs",
            correlation_result.get("highly_correlated_pairs", [])
        )
        n_pairs = correlation_result.get("n_high_pairs", len(high_corr_pairs))
        if n_pairs >= 3: uniqueness -= 30
        elif n_pairs >= 1: uniqueness -= 15

    validity = 100
    if outlier_result is not None and outlier_result.get("supported", True):
        outlier_pct = outlier_result.get(
            "total_outlier_pct",
            outlier_result.get("outlier_row_percentage", 0)
        )
        if outlier_pct >= 15: validity -= 30
        elif outlier_pct >= 5: validity -= 15
        elif outlier_pct > 0: validity -= 5
    if imbalance_result is not None and imbalance_result.get("supported"):
        if imbalance_result.get("is_imbalanced"): validity -= 20

    category_scores = {
        "Completeness": max(0, completeness),
        "Consistency": max(0, consistency),
        "Accuracy": max(0, accuracy),
        "Uniqueness": max(0, uniqueness),
        "Validity": max(0, validity)
    }

    return {
        "score": score,
        "status": status,
        "issues": issues,
        "category_scores": category_scores
    }