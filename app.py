"""
Data Quality Analyzer — Streamlit Dashboard

A professional-grade data quality analysis tool with interactive
visualizations, multi-check analysis, and downloadable reports.

Run with:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import streamlit as st

# --- Quality checks ---
from checks.profiling import profile_dataset, summarize_profile
from checks.label_noise import detect_label_noise, summarize_label_noise
from checks.leakage import detect_leakage, summarize_leakage
from checks.duplicates import detect_duplicates, summarize_duplicates
from checks.imbalance import detect_class_imbalance, summarize_class_imbalance
from checks.outlier_detection import detect_outliers, summarize_outliers
from checks.correlation import detect_high_correlation, summarize_correlation
from checks.data_type_issues import detect_data_type_issues, summarize_data_type_issues

# --- Scoring & gate ---
from report.readiness_score import calculate_readiness_score
from pipeline.quality_gate import apply_quality_gate

# --- Visualizations ---
from report.visualizations import (
    create_readiness_gauge,
    create_missing_values_chart,
    create_correlation_heatmap,
    create_class_distribution_chart,
    create_leakage_chart,
    create_outlier_box_plots,
    create_quality_radar_chart,
    create_data_types_chart,
)

# --- Report generation ---
from report.generate_report import generate_report, generate_report_html


# ==================================================
# PAGE CONFIG & CUSTOM THEME
# ==================================================

st.set_page_config(
    page_title="Data Quality Analyzer",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        color: white !important;
        margin: 0 !important;
        font-size: 2rem !important;
    }
    .main-header p {
        color: rgba(255,255,255,0.85) !important;
        margin: 0.3rem 0 0 0 !important;
        font-size: 1rem;
    }

    /* Score card styling */
    .score-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .score-card.pass { border-left: 5px solid #28a745; }
    .score-card.warning { border-left: 5px solid #ffc107; }
    .score-card.fail { border-left: 5px solid #dc3545; }

    /* Check status badges */
    .badge-pass {
        background-color: #d4edda; color: #155724;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600;
    }
    .badge-warn {
        background-color: #fff3cd; color: #856404;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600;
    }
    .badge-fail {
        background-color: #f8d7da; color: #721c24;
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.85rem; font-weight: 600;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown { color: #e0e0e0; }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        padding: 12px;
        border-radius: 8px;
    }

    /* Section dividers */
    .section-header {
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ==================================================
# SESSION STATE INITIALIZATION
# ==================================================

if "df" not in st.session_state:
    st.session_state.df = None
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None
if "report_md" not in st.session_state:
    st.session_state.report_md = None


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## 🧪 Data Quality Analyzer")
    st.markdown("---")

    # File uploader in sidebar
    uploaded_file = st.file_uploader(
        "📁 Upload CSV Dataset",
        type=["csv"],
        help="Upload a CSV file to analyze its quality",
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.success(f"✅ Loaded: {uploaded_file.name}")
        st.markdown(f"**Rows:** {len(df):,}  |  **Columns:** {len(df.columns)}")

        st.markdown("---")

        # Target column selector
        target_col = st.selectbox(
            "🎯 Target Column",
            options=df.columns,
            index=len(df.columns) - 1,
            help="The column you're trying to predict (for ML-specific checks)",
        )

        st.markdown("---")

        # Navigation
        st.markdown("### 📑 Navigation")
        page = st.radio(
            "Go to:",
            [
                "🏠 Overview & Profile",
                "🚀 Full Analysis",
                "🔬 Individual Checks",
                "📊 Report & Export",
            ],
            label_visibility="collapsed",
        )
    else:
        page = None
        target_col = None
        df = None

    st.markdown("---")
    st.caption("v2.0 — Portfolio Edition")


# ==================================================
# MAIN CONTENT
# ==================================================

# --- Header ---
st.markdown("""
<div class="main-header">
    <h1>🧪 Data Quality Analyzer</h1>
    <p>Automated data quality assessment for machine learning pipelines</p>
</div>
""", unsafe_allow_html=True)


if df is None:
    # Landing page when no file is uploaded
    st.markdown("## Welcome!")
    st.markdown(
        "Upload a CSV dataset using the sidebar to get started. "
        "This tool will analyze your data across **8 quality dimensions** "
        "and produce a readiness score for ML training."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("#### 📋 Profiling\nMissing values, types, duplicates")
    col2.markdown("#### 🔍 Noise Detection\nMislabeled rows via cross-validation")
    col3.markdown("#### 🚨 Leakage Detection\nFeatures that leak the target")
    col4.markdown("#### 📈 Outlier Analysis\nStatistical anomaly detection")

    col5, col6, col7, col8 = st.columns(4)
    col5.markdown("#### ⚖️ Class Imbalance\nTarget distribution analysis")
    col6.markdown("#### 🔁 Duplicate Detection\nExact row duplication")
    col7.markdown("#### 🔗 Correlation Analysis\nRedundant feature detection")
    col8.markdown("#### 🏷️ Type Validation\nData type consistency checks")

    st.info("💡 Don't have a dataset? Try the sample data in `sample_data/sample_customers.csv`")


# ==================================================
# PAGE: OVERVIEW & PROFILE
# ==================================================

elif page == "🏠 Overview & Profile":

    st.markdown("## 📋 Dataset Overview")

    profile = profile_dataset(df)

    # --- Key metrics row ---
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{profile['n_rows']:,}")
    m2.metric("Columns", f"{profile['n_cols']}")
    m3.metric("Duplicates", f"{profile['n_duplicates']} ({profile['duplicate_pct']}%)")
    m4.metric("Constant Cols", f"{len(profile['constant_cols'])}")

    missing_total = profile["missing_report"]["missing_count"].sum() if not profile["missing_report"].empty else 0
    total_cells = profile["n_rows"] * profile["n_cols"]
    missing_overall_pct = round(missing_total / total_cells * 100, 2) if total_cells > 0 else 0
    m5.metric("Overall Missing", f"{missing_overall_pct}%")

    st.markdown("---")

    # --- Two-column layout: Data types + Missing values ---
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🏷️ Data Types Distribution")
        dtype_counts = profile["dtypes_report"].value_counts()
        fig_types = create_data_types_chart(dtype_counts)
        st.plotly_chart(fig_types, use_container_width=True)

    with col_right:
        st.markdown("### ❌ Missing Values")
        if not profile["missing_report"].empty:
            fig_missing = create_missing_values_chart(profile["missing_report"])
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ No missing values found!")

    # --- Data preview ---
    st.markdown("---")
    st.markdown("### 👀 Data Preview")
    st.dataframe(df.head(20), use_container_width=True, height=400)

    # --- Key findings ---
    st.markdown("### 📝 Key Findings")
    for line in summarize_profile(profile):
        st.write(f"• {line}")

    if profile["constant_cols"]:
        st.warning(
            f"⚠️ Constant columns detected (carry no information): "
            f"`{'`, `'.join(profile['constant_cols'])}`"
        )

    # --- Cardinality ---
    if profile["cardinality"]:
        with st.expander("📊 Categorical Column Cardinality"):
            card_df = pd.DataFrame.from_dict(
                profile["cardinality"], orient="index", columns=["Unique Values"]
            ).sort_values("Unique Values", ascending=False)
            st.dataframe(card_df, use_container_width=True)


# ==================================================
# PAGE: FULL ANALYSIS
# ==================================================

elif page == "🚀 Full Analysis":

    st.markdown("## 🚀 Complete Dataset Analysis")
    st.markdown(
        "Run all **8 quality checks** at once and get a unified readiness score "
        "with an interactive quality report."
    )

    if st.button("🚀 Run Complete Analysis", type="primary", use_container_width=True):

        results = {}
        progress = st.progress(0, text="Starting analysis...")

        # Step 1: Basic profiling
        progress.progress(5, text="📋 Running basic profiling...")
        results["profile"] = profile_dataset(df)

        # Step 2: Duplicate detection
        progress.progress(15, text="🔁 Detecting duplicates...")
        results["duplicates"] = detect_duplicates(df)

        # Step 3: Outlier detection
        progress.progress(25, text="📈 Detecting outliers...")
        results["outliers"] = detect_outliers(df)

        # Step 4: Correlation analysis
        progress.progress(35, text="🔗 Analyzing correlations...")
        results["correlation"] = detect_high_correlation(df)

        # Step 5: Data type issues
        progress.progress(45, text="🏷️ Checking data types...")
        results["type_issues"] = detect_data_type_issues(df)

        # Step 6: Class imbalance
        progress.progress(55, text="⚖️ Checking class imbalance...")
        results["imbalance"] = detect_class_imbalance(df, target_col=target_col)

        # Step 7: Label noise
        progress.progress(70, text="🔍 Detecting label noise (training model)...")
        results["label_noise"] = detect_label_noise(df, target_col=target_col)

        # Step 8: Leakage
        progress.progress(85, text="🚨 Detecting feature leakage...")
        results["leakage"] = detect_leakage(df, target_col=target_col)

        # Step 9: Calculate scores
        progress.progress(95, text="🏆 Calculating readiness score...")
        results["readiness"] = calculate_readiness_score(
            profile=results["profile"],
            duplicate_result=results["duplicates"],
            imbalance_result=results["imbalance"],
            label_noise_result=results["label_noise"],
            leakage_result=results["leakage"],
            outlier_result=results.get("outliers"),
            correlation_result=results.get("correlation"),
            type_issues_result=results.get("type_issues"),
        )

        results["quality_gate"] = apply_quality_gate(results["readiness"])

        # Generate report
        report_md = generate_report(
            profile=results["profile"],
            readiness_result=results["readiness"],
            quality_gate_result=results["quality_gate"],
            label_noise_result=results["label_noise"],
            leakage_result=results["leakage"],
            imbalance_result=results["imbalance"],
            duplicate_result=results["duplicates"],
            outlier_result=results.get("outliers"),
            correlation_result=results.get("correlation"),
            type_issues_result=results.get("type_issues"),
        )

        progress.progress(100, text="✅ Analysis complete!")
        st.session_state.analysis_results = results
        st.session_state.report_md = report_md

    # --- Display results if available ---
    if st.session_state.analysis_results is not None:
        results = st.session_state.analysis_results
        score = results["readiness"]["score"]
        status = results["readiness"]["status"]
        decision = results["quality_gate"]["decision"]

        st.markdown("---")

        # === READINESS SCORE SECTION ===
        gauge_col, info_col = st.columns([1, 1])

        with gauge_col:
            st.markdown("### 🏆 Dataset Readiness Score")
            fig_gauge = create_readiness_gauge(score, status)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with info_col:
            st.markdown("### 🚦 Quality Gate Decision")
            st.markdown("<br>", unsafe_allow_html=True)

            if decision == "APPROVED":
                st.success(f"✅ **{decision}** — {results['quality_gate']['message']}")
            elif decision == "REVIEW REQUIRED":
                st.warning(f"⚠️ **{decision}** — {results['quality_gate']['message']}")
            else:
                st.error(f"❌ **{decision}** — {results['quality_gate']['message']}")

            st.markdown("<br>", unsafe_allow_html=True)

            # Issues list
            if results["readiness"]["issues"]:
                st.markdown("**Issues Found:**")
                for issue in results["readiness"]["issues"]:
                    st.write(f"• {issue}")
            else:
                st.success("🎉 No major data quality issues detected!")

        st.markdown("---")

        # === RADAR CHART (Quality Dimensions) ===
        if "category_scores" in results["readiness"]:
            st.markdown("### 🕸️ Quality Dimensions Breakdown")
            fig_radar = create_quality_radar_chart(results["readiness"]["category_scores"])
            st.plotly_chart(fig_radar, use_container_width=True)
            st.markdown("---")

        # === CHECK SUMMARIES GRID ===
        st.markdown("### 📊 Individual Check Results")

        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

        with row1_col1:
            st.markdown("**🔁 Duplicates**")
            st.write(summarize_duplicates(results["duplicates"]))

        with row1_col2:
            st.markdown("**📈 Outliers**")
            if results["outliers"].get("supported", True):
                st.write(summarize_outliers(results["outliers"]))
            else:
                st.info("No numeric columns to check.")

        with row1_col3:
            st.markdown("**🔗 Correlation**")
            st.write(summarize_correlation(results["correlation"]))

        with row1_col4:
            st.markdown("**🏷️ Type Issues**")
            st.write(summarize_data_type_issues(results["type_issues"]))

        row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)

        with row2_col1:
            st.markdown("**⚖️ Class Imbalance**")
            st.write(summarize_class_imbalance(results["imbalance"]))

        with row2_col2:
            st.markdown("**🔍 Label Noise**")
            st.write(summarize_label_noise(results["label_noise"]))

        with row2_col3:
            st.markdown("**🚨 Leakage**")
            st.write(summarize_leakage(results["leakage"]))

        with row2_col4:
            missing_cols = len(results["profile"]["missing_report"])
            if missing_cols > 0:
                st.markdown("**❌ Missing Values**")
                st.write(f"{missing_cols} column(s) have missing data.")
            else:
                st.markdown("**❌ Missing Values**")
                st.write("No missing values found.")

        st.markdown("---")

        # === INTERACTIVE CHARTS ===
        st.markdown("### 📈 Interactive Visualizations")

        viz_tabs = st.tabs([
            "Class Distribution",
            "Leakage Scores",
            "Outlier Box Plots",
            "Correlation Heatmap",
            "Missing Values",
        ])

        with viz_tabs[0]:
            if results["imbalance"].get("supported"):
                fig_class = create_class_distribution_chart(results["imbalance"]["distribution"])
                st.plotly_chart(fig_class, use_container_width=True)
            else:
                st.info(summarize_class_imbalance(results["imbalance"]))

        with viz_tabs[1]:
            if results["leakage"].get("supported") and len(results["leakage"]["all_scores"]) > 0:
                fig_leak = create_leakage_chart(
                    results["leakage"]["all_scores"],
                    results["leakage"]["suspicion_threshold"],
                )
                st.plotly_chart(fig_leak, use_container_width=True)
            else:
                st.info(summarize_leakage(results["leakage"]))

        with viz_tabs[2]:
            outlier_res = results["outliers"]
            if outlier_res.get("supported", True) and outlier_res.get("total_outlier_rows", 0) > 0:
                outlier_cols = list(outlier_res.get("outlier_indices", {}).keys())
                if outlier_cols:
                    fig_box = create_outlier_box_plots(df, outlier_cols)
                    st.plotly_chart(fig_box, use_container_width=True)
                else:
                    st.success("No outliers detected.")
            else:
                st.success("No outliers detected.")

        with viz_tabs[3]:
            if results["correlation"].get("supported") and results["correlation"].get("correlation_matrix") is not None:
                fig_corr = create_correlation_heatmap(results["correlation"]["correlation_matrix"])
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info(summarize_correlation(results["correlation"]))

        with viz_tabs[4]:
            if not results["profile"]["missing_report"].empty:
                fig_miss = create_missing_values_chart(results["profile"]["missing_report"])
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ No missing values!")


# ==================================================
# PAGE: INDIVIDUAL CHECKS
# ==================================================

elif page == "🔬 Individual Checks":

    st.markdown("## 🔬 Individual Quality Checks")
    st.markdown("Run checks one at a time for detailed analysis.")

    check_tab = st.tabs([
        "📋 Profiling",
        "🔍 Label Noise",
        "🚨 Leakage",
        "⚖️ Imbalance",
        "🔁 Duplicates",
        "📈 Outliers",
        "🔗 Correlation",
        "🏷️ Type Issues",
    ])

    # --- Tab 1: Profiling ---
    with check_tab[0]:
        st.markdown("### 📋 Basic Data Profiling")
        profile = profile_dataset(df)

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", f"{profile['n_rows']:,}")
        col2.metric("Columns", f"{profile['n_cols']}")
        col3.metric("Duplicates", f"{profile['n_duplicates']} ({profile['duplicate_pct']}%)")

        for line in summarize_profile(profile):
            st.write(f"• {line}")

        if not profile["missing_report"].empty:
            st.markdown("**Missing Values by Column:**")
            fig_missing = create_missing_values_chart(profile["missing_report"])
            st.plotly_chart(fig_missing, use_container_width=True)

        st.markdown("**Data Types:**")
        fig_types = create_data_types_chart(profile["dtypes_report"].value_counts())
        st.plotly_chart(fig_types, use_container_width=True)

    # --- Tab 2: Label Noise ---
    with check_tab[1]:
        st.markdown("### 🔍 Label Noise Detection")
        st.caption(
            "Trains a model with cross-validation and flags rows where "
            "the model confidently disagrees with the label."
        )

        if st.button("Run Label Noise Check", key="ln_btn"):
            with st.spinner("Training model with cross-validation..."):
                result = detect_label_noise(df, target_col=target_col)

            if not result["supported"]:
                st.info(summarize_label_noise(result))
            else:
                st.write(summarize_label_noise(result))
                if result["n_flagged"] > 0:
                    st.markdown("**Most suspicious rows:**")
                    st.dataframe(result["flagged_rows"], use_container_width=True)

    # --- Tab 3: Leakage ---
    with check_tab[2]:
        st.markdown("### 🚨 Feature Leakage Detection")
        st.caption(
            "Tests each feature individually — any feature that predicts the target "
            "with suspiciously high accuracy on its own is flagged."
        )

        if st.button("Run Leakage Check", key="leak_btn"):
            with st.spinner("Testing each feature individually..."):
                leak_result = detect_leakage(df, target_col=target_col)

            if not leak_result["supported"]:
                st.info(summarize_leakage(leak_result))
            else:
                if leak_result["n_flagged"] > 0:
                    st.error(summarize_leakage(leak_result))
                else:
                    st.success(summarize_leakage(leak_result))

                fig_leak = create_leakage_chart(
                    leak_result["all_scores"],
                    leak_result["suspicion_threshold"],
                )
                st.plotly_chart(fig_leak, use_container_width=True)

    # --- Tab 4: Imbalance ---
    with check_tab[3]:
        st.markdown("### ⚖️ Class Imbalance Detection")
        st.caption("Checks whether some target classes have significantly fewer examples.")

        if st.button("Run Imbalance Check", key="imb_btn"):
            imb_result = detect_class_imbalance(df, target_col=target_col)

            if not imb_result["supported"]:
                st.info(summarize_class_imbalance(imb_result))
            else:
                if imb_result["is_imbalanced"]:
                    st.warning(summarize_class_imbalance(imb_result))
                else:
                    st.success(summarize_class_imbalance(imb_result))

                fig_class = create_class_distribution_chart(imb_result["distribution"])
                st.plotly_chart(fig_class, use_container_width=True)

    # --- Tab 5: Duplicates ---
    with check_tab[4]:
        st.markdown("### 🔁 Duplicate Detection")
        st.caption("Checks for exact duplicate rows in the dataset.")

        if st.button("Run Duplicate Check", key="dup_btn"):
            dup_result = detect_duplicates(df)

            if dup_result["n_duplicate_rows"] > 0:
                st.warning(summarize_duplicates(dup_result))
                st.markdown("**Duplicate Rows:**")
                st.dataframe(dup_result["duplicate_rows"], use_container_width=True)
            else:
                st.success(summarize_duplicates(dup_result))

    # --- Tab 6: Outliers ---
    with check_tab[5]:
        st.markdown("### 📈 Outlier Detection")
        st.caption("Detects statistical outliers using IQR or Z-score methods.")

        method = st.radio("Method:", ["iqr", "zscore"], horizontal=True, key="outlier_method")
        threshold = st.slider(
            "Threshold:",
            min_value=1.0, max_value=5.0,
            value=1.5 if method == "iqr" else 3.0,
            step=0.5,
            key="outlier_thresh",
        )

        if st.button("Run Outlier Check", key="out_btn"):
            out_result = detect_outliers(df, method=method, threshold=threshold)

            st.write(summarize_outliers(out_result))

            if out_result.get("total_outlier_rows", 0) > 0:
                st.markdown("**Outlier Summary per Column:**")
                st.dataframe(out_result["outlier_summary"], use_container_width=True)

                outlier_cols = list(out_result.get("outlier_indices", {}).keys())
                if outlier_cols:
                    fig_box = create_outlier_box_plots(df, outlier_cols)
                    st.plotly_chart(fig_box, use_container_width=True)

    # --- Tab 7: Correlation ---
    with check_tab[6]:
        st.markdown("### 🔗 Feature Correlation Analysis")
        st.caption("Detects highly correlated feature pairs that may be redundant.")

        corr_threshold = st.slider(
            "Correlation threshold:",
            min_value=0.5, max_value=1.0,
            value=0.95, step=0.05,
            key="corr_thresh",
        )

        if st.button("Run Correlation Check", key="corr_btn"):
            corr_result = detect_high_correlation(df, threshold=corr_threshold)

            if not corr_result.get("supported"):
                st.info(summarize_correlation(corr_result))
            else:
                st.write(summarize_correlation(corr_result))

                if corr_result.get("correlation_matrix") is not None:
                    fig_corr = create_correlation_heatmap(corr_result["correlation_matrix"])
                    st.plotly_chart(fig_corr, use_container_width=True)

                if corr_result["n_high_pairs"] > 0:
                    st.markdown("**Highly correlated pairs:**")
                    pairs_df = pd.DataFrame(
                        corr_result["high_correlation_pairs"],
                        columns=["Feature 1", "Feature 2", "Correlation"],
                    )
                    st.dataframe(pairs_df, use_container_width=True)

    # --- Tab 8: Type Issues ---
    with check_tab[7]:
        st.markdown("### 🏷️ Data Type Issue Detection")
        st.caption(
            "Finds columns with mixed types, numbers stored as strings, "
            "and inconsistent categorical encoding."
        )

        if st.button("Run Type Check", key="type_btn"):
            type_result = detect_data_type_issues(df)

            st.write(summarize_data_type_issues(type_result))

            if type_result.get("numeric_as_string"):
                st.warning(
                    f"Numeric columns stored as strings: "
                    f"`{'`, `'.join(type_result['numeric_as_string'])}`"
                )

            if type_result.get("inconsistent_casing"):
                st.markdown("**Inconsistent Casing:**")
                for col, variants in type_result["inconsistent_casing"].items():
                    st.write(f"• **{col}**: {variants}")

            if type_result.get("mixed_types"):
                st.warning(
                    f"Columns with mixed types: "
                    f"`{'`, `'.join(type_result['mixed_types'])}`"
                )


# ==================================================
# PAGE: REPORT & EXPORT
# ==================================================

elif page == "📊 Report & Export":

    st.markdown("## 📊 Analysis Report & Export")

    if st.session_state.analysis_results is None:
        st.info(
            "⚡ Run a **Full Analysis** first (go to 🚀 Full Analysis page), "
            "then come back here to view and download the report."
        )
    else:
        results = st.session_state.analysis_results
        report_md = st.session_state.report_md

        # --- Display the report ---
        st.markdown("### 📄 Full Analysis Report")
        st.markdown(report_md)

        st.markdown("---")

        # --- Download buttons ---
        st.markdown("### 📥 Download Report")

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            st.download_button(
                label="📥 Download as Markdown",
                data=report_md,
                file_name="data_quality_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with dl_col2:
            report_html = generate_report_html(report_md)
            st.download_button(
                label="📥 Download as HTML",
                data=report_html,
                file_name="data_quality_report.html",
                mime="text/html",
                use_container_width=True,
            )

        with dl_col3:
            # Export flagged rows if label noise was detected
            if (
                results["label_noise"].get("supported")
                and results["label_noise"]["n_flagged"] > 0
            ):
                flagged_csv = results["label_noise"]["flagged_rows"].to_csv(index=False)
                st.download_button(
                    label="📥 Download Flagged Rows (CSV)",
                    data=flagged_csv,
                    file_name="flagged_label_noise_rows.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.button(
                    "📥 No Flagged Rows",
                    disabled=True,
                    use_container_width=True,
                )