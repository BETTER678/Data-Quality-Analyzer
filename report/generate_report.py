"""
Report Generation for Data Quality Analyzer

Generates plain-English markdown and styled HTML reports summarizing all
data quality findings, readiness scores, and actionable recommendations.
"""

from typing import Optional


def generate_report(
    profile: dict,
    readiness_result: dict,
    quality_gate_result: dict,
    label_noise_result: Optional[dict] = None,
    leakage_result: Optional[dict] = None,
    imbalance_result: Optional[dict] = None,
    duplicate_result: Optional[dict] = None,
    outlier_result: Optional[dict] = None,
    correlation_result: Optional[dict] = None,
    type_issues_result: Optional[dict] = None,
) -> str:
    """
    Generate a comprehensive plain-English Markdown report summarizing all findings.
    """
    score = readiness_result.get("score", 0)
    status = readiness_result.get("status", "FAIL")
    decision = quality_gate_result.get("decision", "BLOCKED")
    gate_message = quality_gate_result.get("message", "")

    report = []

    # Title & Metadata
    report.append("# 🧪 Data Quality Assessment Report")
    report.append("")
    report.append("---")
    report.append("")

    # Executive Summary
    report.append("## 1. Executive Summary")
    report.append(f"- **Overall Readiness Score:** `{score}/100` ({status})")
    report.append(f"- **Quality Gate Ruling:** **{decision}**")
    report.append(f"- **Gate Evaluation:** {gate_message}")

    if score >= 80:
        verdict_text = (
            "The dataset meets production quality standards and is certified for ML model training. "
            "Minor observations noted below should still be monitored during feature engineering."
        )
    elif score >= 60:
        verdict_text = (
            "The dataset exhibits moderate quality anomalies. While usable for baseline experimentation, "
            "the highlighted issues must be remediated before training production models."
        )
    else:
        verdict_text = (
            "CRITICAL: The dataset has failed the quality gate. Proceeding with model training on this "
            "data risks severe degradation, silent bias, and inaccurate inferences. Rectify blockers below."
        )
    report.append(f"- **Summary Assessment:** {verdict_text}")
    report.append("")

    # Key Findings
    report.append("## 2. Key Findings & Issues")
    issues = readiness_result.get("issues", [])
    if issues:
        for idx, issue in enumerate(issues, 1):
            report.append(f"{idx}. {issue}")
    else:
        report.append("✅ No critical data quality defects were detected.")
    report.append("")

    # Detailed Analysis
    report.append("## 3. Detailed Dimension Analysis")

    # Dataset Profile
    if profile:
        report.append("### 📋 Profiling & Completeness")
        report.append(f"- **Total Records:** {profile.get('n_rows', 0):,}")
        report.append(f"- **Total Features:** {profile.get('n_cols', 0)}")
        missing_df = profile.get("missing_report")
        if missing_df is not None and not missing_df.empty:
            worst_col = missing_df.index[0]
            worst_pct = missing_df.iloc[0]["missing_pct"]
            report.append(f"- **Missing Values:** {len(missing_df)} column(s) affected (worst: `{worst_col}` at {worst_pct}%)")
        else:
            report.append("- **Missing Values:** 0% missing across all columns.")

        constant_cols = profile.get("constant_cols", [])
        if constant_cols:
            report.append(f"- **Constant Columns (Zero Variance):** `{', '.join(constant_cols)}`")
        else:
            report.append("- **Constant Columns:** None detected.")
        report.append("")

    # Duplicates
    if duplicate_result:
        report.append("### 🔁 Record Uniqueness")
        report.append(f"- **Duplicate Rows:** {duplicate_result.get('n_duplicate_rows', 0)} ({duplicate_result.get('duplicate_percentage', 0):.2f}%)")
        report.append("")

    # Outliers
    if outlier_result and outlier_result.get("supported"):
        report.append("### 📈 Statistical Outliers")
        method = outlier_result.get("method", "IQR").upper()
        n_outliers = outlier_result.get("total_outlier_rows", 0)
        pct_outliers = outlier_result.get("total_outlier_pct", 0)
        report.append(f"- **Method:** {method}")
        report.append(f"- **Flagged Outlier Records:** {n_outliers} ({pct_outliers:.2f}%)")
        report.append("")

    # Correlation
    if correlation_result and correlation_result.get("supported"):
        report.append("### 🔗 Feature Multicollinearity")
        n_pairs = correlation_result.get("n_high_pairs", 0)
        report.append(f"- **Redundant High-Correlation Pairs (|r| >= {correlation_result.get('threshold', 0.95)}):** {n_pairs}")
        for p1, p2, r in correlation_result.get("high_correlation_pairs", [])[:5]:
            report.append(f"  - `{p1}` <-> `{p2}`: r = {r:.3f}")
        report.append("")

    # Data Types
    if type_issues_result and type_issues_result.get("supported"):
        report.append("### 🏷️ Schema & Type Integrity")
        report.append(f"- **Identified Type Inconsistencies:** {type_issues_result.get('n_issues', 0)}")
        if type_issues_result.get("numeric_as_string"):
            report.append(f"  - Numeric stored as string: `{', '.join(type_issues_result['numeric_as_string'])}`")
        if type_issues_result.get("mixed_types"):
            report.append(f"  - Mixed data types: `{', '.join(type_issues_result['mixed_types'])}`")
        report.append("")

    # ML Specific Checks
    if (label_noise_result and label_noise_result.get("supported")) or \
       (leakage_result and leakage_result.get("supported")) or \
       (imbalance_result and imbalance_result.get("supported")):
        report.append("### 🎯 Machine Learning Readiness")

        if label_noise_result and label_noise_result.get("supported"):
            n_noise = label_noise_result.get("n_flagged", 0)
            noise_pct = label_noise_result.get("flagged_pct", 0)
            report.append(f"- **Suspicious Label Noise:** {n_noise} rows ({noise_pct:.2f}%) where model cross-validation strongly contradicted actual label.")

        if leakage_result and leakage_result.get("supported"):
            n_leak = leakage_result.get("n_flagged", 0)
            report.append(f"- **Potential Target Leakage:** {n_leak} feature(s) predict target with suspicious solo accuracy.")
            if n_leak > 0:
                top_leak = leakage_result["flagged_features"].iloc[0]
                report.append(f"  - Top suspect: `{top_leak['feature']}` ({top_leak['solo_accuracy']*100:.1f}% solo accuracy)")

        if imbalance_result and imbalance_result.get("supported"):
            is_imb = imbalance_result.get("is_imbalanced", False)
            min_pct = imbalance_result.get("minority_percentage", 0)
            report.append(f"- **Class Distribution:** {'Imbalanced' if is_imb else 'Balanced'} (Minority class: {min_pct:.1f}%)")

        report.append("")

    # Recommendations
    report.append("## 4. Remediation Action Plan")
    rec_num = 1
    if duplicate_result and duplicate_result.get("n_duplicate_rows", 0) > 0:
        report.append(f"{rec_num}. **Deduplicate Records:** Drop {duplicate_result['n_duplicate_rows']} duplicate rows to prevent overfitting.")
        rec_num += 1

    if profile and profile.get("constant_cols"):
        report.append(f"{rec_num}. **Drop Zero-Variance Features:** Remove `{', '.join(profile['constant_cols'])}` as they provide zero information entropy.")
        rec_num += 1

    if profile and profile.get("missing_report") is not None and not profile["missing_report"].empty:
        report.append(f"{rec_num}. **Imputation Strategy:** Apply median/mode or model-based imputation to incomplete fields.")
        rec_num += 1

    if leakage_result and leakage_result.get("supported") and leakage_result.get("n_flagged", 0) > 0:
        report.append(f"{rec_num}. **Isolate Leaking Features:** Audit and exclude features flagged with near-deterministic solo predictive power.")
        rec_num += 1

    if label_noise_result and label_noise_result.get("supported") and label_noise_result.get("n_flagged", 0) > 0:
        report.append(f"{rec_num}. **Relabel or Prune Flagged Samples:** Inspect flagged label noise instances for human annotation errors.")
        rec_num += 1

    if correlation_result and correlation_result.get("supported") and correlation_result.get("n_high_pairs", 0) > 0:
        report.append(f"{rec_num}. **Eliminate Collinearity:** Drop one variable from each redundant feature pair (|r| >= {correlation_result.get('threshold', 0.95)}).")
        rec_num += 1

    if rec_num == 1:
        report.append("1. **Continuous Monitoring:** Dataset is clean. Track drift across future data batches.")

    report.append("")
    report.append("---")
    report.append("*Generated by Data Quality Analyzer (v2.0 Portfolio Edition)*")

    return "\n".join(report)


def generate_report_html(markdown_report: str) -> str:
    """
    Convert the markdown report into a standalone, styled HTML document.
    """
    lines = markdown_report.split("\n")
    html_parts = []

    in_list = False
    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue

        if line.startswith("# "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("- "):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            content = line[2:]
            # format bold
            content = _format_inline_markdown(content)
            html_parts.append(f"<li>{content}</li>")
        elif line.startswith("---"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            html_parts.append("<hr/>")
        elif line[0].isdigit() and len(line) > 2 and line[1] in [".", ")"]:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            content = _format_inline_markdown(line[3:])
            html_parts.append(f"<p class='ordered-item'><strong>{line[:2]}</strong> {content}</p>")
        else:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            content = _format_inline_markdown(line)
            html_parts.append(f"<p>{content}</p>")

    if in_list:
        html_parts.append("</ul>")

    body_content = "\n".join(html_parts)

    html_document = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Quality Assessment Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #1f2937;
            background-color: #f9fafb;
            padding: 40px 20px;
            margin: 0;
        }}
        .container {{
            max-width: 860px;
            margin: 0 auto;
            background: #ffffff;
            padding: 40px 50px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            border: 1px solid #e5e7eb;
        }}
        h1 {{
            color: #4338ca;
            font-size: 2rem;
            margin-top: 0;
            padding-bottom: 12px;
            border-bottom: 2px solid #e0e7ff;
        }}
        h2 {{
            color: #1e293b;
            font-size: 1.35rem;
            margin-top: 28px;
            padding-bottom: 6px;
            border-bottom: 1px solid #f1f5f9;
        }}
        h3 {{
            color: #334155;
            font-size: 1.1rem;
            margin-top: 18px;
        }}
        ul {{
            padding-left: 24px;
        }}
        li {{
            margin-bottom: 6px;
        }}
        code {{
            background-color: #f1f5f9;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.9em;
            color: #0f172a;
        }}
        hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 24px 0;
        }}
        .ordered-item {{
            margin-left: 10px;
            margin-bottom: 8px;
        }}
        @media print {{
            body {{ background: #fff; padding: 0; }}
            .container {{ box-shadow: none; border: none; padding: 0; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        {body_content}
    </div>
</body>
</html>
"""
    return html_document


def _format_inline_markdown(text: str) -> str:
    """Helper to convert **bold** and `code` tags."""
    import re
    # Bold
    text = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", text)
    # Code
    text = re.sub(r"`(.*?)`", r"<code>\1</code>", text)
    return text
