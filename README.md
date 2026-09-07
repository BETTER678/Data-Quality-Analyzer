# 🧪 Data Quality Analyzer (v2.0 Portfolio Edition)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63+-FF4B4B.svg?logo=streamlit)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75.svg?logo=plotly)](https://plotly.com/)
[![Tests](https://img.shields.io/badge/tests-40%20passed-success.svg)](https://docs.pytest.org/)
[![Docker Ready](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, automated data quality evaluation and governance framework designed to audit tabular datasets before they enter machine learning training pipelines. It blocks corrupt or leaking data via a deterministic **Quality Gate** and evaluates datasets across an aggregated **Readiness Score (0-100)** spanning 5 core data quality dimensions.

---

## 🌟 Why This Project Stands Out (2026 Portfolio Highlights)

- **8 Multi-Faceted Quality Checks**: Goes far beyond trivial null checks to detect subtle ML-breaking defects like out-of-fold label noise, target leakage, multicollinearity, and split contamination.
- **5-Dimensional Quality Radar**: Scores datasets on *Completeness*, *Consistency*, *Accuracy*, *Uniqueness*, and *Validity*.
- **Modern Interactive Visualizations**: Features animated Plotly gauges, radar charts, solo predictive power leakage bars, outlier box plots, and correlation heatmaps.
- **Dual-Interface Architecture**:
  - 🖥️ **Streamlit Web Dashboard**: Interactive UI with sidebar navigation, real-time analytics, session persistence, and downloadable HTML/Markdown reports.
  - ⚡ **FastAPI REST API**: High-throughput programmatic backend (`POST /analyze`, `GET /health`, `GET /checks`) ready for CI/CD or MLOps pipeline integration.
- **Robust Automated Test Suite**: 40 unit and integration tests written in `pytest` covering all checks, scoring math, gate boundaries, and API endpoints.
- **Container Ready**: Includes multi-stage `Dockerfile` and `docker-compose.yml` to orchestrate both the API service and Dashboard without local environment conflicts.
- **100% Free & Local**: Zero external API dependencies, zero cloud subscriptions required.

---

## 🔬 The 8 Data Quality Checks

| Check | Module | What It Detects | Real-World Impact |
|---|---|---|---|
| **1. Basic Profiling** | `checks/profiling.py` | Missing values, single-value constant columns, data types, cardinality | Prevents zero-variance features and unhandled null crashes |
| **2. Label Noise Detection** | `checks/label_noise.py` | Out-of-fold cross-validated misclassifications with high model confidence | Prevents models from memorizing mislabeled data annotations |
| **3. Data Leakage Detection** | `checks/leakage.py` | Trains single-feature models to catch features with suspicious solo accuracy | Catches post-event variables that falsely inflate validation metrics |
| **4. Train/Test Contamination** | `checks/duplicates.py` | Exact duplicates and overlap across train/test splits | Prevents over-optimistic evaluation metrics from data snooping |
| **5. Class Imbalance** | `checks/imbalance.py` | Target class distribution and minority ratio | Flags severe skew needing SMOTE, focal loss, or stratified sampling |
| **6. Statistical Outliers** | `checks/outlier_detection.py` | IQR (Interquartile Range) and Z-score anomaly detection | Prevents linear models and distance metrics from distortion |
| **7. Feature Correlation** | `checks/correlation.py` | Pearson correlation matrix with threshold detection (\|r\| ≥ 0.95) | Eliminates multicollinearity and redundant feature weights |
| **8. Schema & Type Integrity** | `checks/data_type_issues.py` | Numbers stored as strings, mixed dtypes, and inconsistent casing ("Yes"/"yes") | Catches parser bugs and silent category splitting |

---

## 🏗️ System Architecture

```
                       ┌────────────────────────┐
                       │   Raw CSV Dataset(s)   │
                       └───────────┬────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼                                         ▼
   ┌───────────────────────┐                 ┌───────────────────────┐
   │  Streamlit Dashboard  │                 │    FastAPI REST API   │
   │      (Port 8501)      │                 │      (Port 8000)      │
   └──────────┬────────────┘                 └──────────┬────────────┘
              │                                         │
              └────────────────────┬────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────┐
              │          8 Core Quality Checks          │
              │  [Profiling, Leakage, Label Noise,      │
              │   Outliers, Correlation, Duplicates,    │
              │   Imbalance, Type Validation]           │
              └────────────────────┬────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────┐
              │     Readiness Scoring Engine (0-100)    │
              │   - Completeness  - Consistency         │
              │   - Accuracy      - Uniqueness          │
              │   - Validity                            │
              └────────────────────┬────────────────────┘
                                   ▼
              ┌─────────────────────────────────────────┐
              │           🚦 Quality Gate               │
              │   - APPROVED        (Score >= 80)       │
              │   - REVIEW REQUIRED (Score 60 - 79)     │
              │   - BLOCKED         (Score < 60)        │
              └────────────────────┬────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
       ┌─────────────────────────┐   ┌─────────────────────────┐
       │   Automated Reports     │   │   Pipeline Decision     │
       │   - Interactive Plotly  │   │   - Block Training      │
       │   - Markdown Summary    │   │   - Promote to Baseline │
       │   - Standalone HTML     │   │   - Alert MLOps Team    │
       └─────────────────────────┘   └─────────────────────────┘
```

---

## 🚦 Quality Gate & Readiness Scoring

Every dataset receives an aggregated **Dataset Readiness Score (0 to 100)**:
- Starts at **100 points** with deterministic penalties deducted per identified defect (e.g. -25 for target leakage, -20 for high label noise, -15 for severe duplicates/outliers).
- Scores are grouped into 5 radar dimensions: **Completeness**, **Consistency**, **Accuracy**, **Uniqueness**, and **Validity**.
- The **Quality Gate** issues an automated governance decision:
  - `APPROVED` (Score ≥ 80): Certified for training pipelines.
  - `REVIEW REQUIRED` (Score 60–79): Usable for prototyping; requires data engineer review.
  - `BLOCKED` (Score < 60): Execution halts immediately to prevent model degradation.

---

## 🚀 Quick Start

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/data-quality-analyzer.git
cd data-quality-analyzer

python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the Interactive Dashboard

```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` and upload `sample_data/sample_customers.csv`.

### 3. Run as a REST API

```bash
uvicorn api.main:app --reload --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Health check: `curl http://localhost:8000/health`
- Programmatic analysis:
  ```bash
  curl -X POST "http://localhost:8000/analyze" \
       -F "file=@sample_data/sample_customers.csv" \
       -F "target_col=churned"
  ```

### 4. Run Automated Test Suite

```bash
pytest tests/ -v
```
Runs all 40 automated test cases verifying checks, scoring, and API endpoints.

---

## 🐳 Running with Docker (Optional)

If you have Docker installed, you can spin up both services with a single command:

```bash
docker-compose up --build
```
- Dashboard: `http://localhost:8501`
- REST API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
data-quality-analyzer/
├── app.py                     # Streamlit web dashboard (interactive UI)
├── requirements.txt           # Project dependencies
├── Dockerfile                 # Multi-purpose container image
├── docker-compose.yml         # Compose configuration for API & UI
├── .dockerignore              # Docker build exclusions
├── README.md                  # Project documentation
├── GETTING_STARTED.md         # Step-by-step tutorial guide
│
├── api/                       # FastAPI REST backend
│   ├── __init__.py
│   └── main.py                # Endpoints: /health, /checks, /analyze
│
├── checks/                    # 8 Modular Data Quality Checks
│   ├── __init__.py
│   ├── profiling.py           # Missing values, constant cols, dtypes
│   ├── label_noise.py         # Out-of-fold model misclassifications
│   ├── leakage.py             # Solo predictive feature leakage
│   ├── duplicates.py          # Exact rows & train/test contamination
│   ├── imbalance.py           # Target distribution skew
│   ├── outlier_detection.py   # Statistical IQR & Z-Score outliers
│   ├── correlation.py         # Pearson correlation & multicollinearity
│   └── data_type_issues.py    # Numbers as strings, mixed types, casing
│
├── pipeline/                  # Governance & Gating
│   └── quality_gate.py        # Automated pass/fail decision rules
│
├── report/                    # Scoring, Visualizations & Reporting
│   ├── readiness_score.py     # Aggregated 0-100 scoring & 5D radar
│   ├── visualizations.py      # Plotly charts (gauges, heatmaps, radar)
│   └── generate_report.py     # Markdown & standalone HTML report generators
│
├── sample_data/               # Test Datasets
│   └── sample_customers.csv   # Injected defects (leakage, noise, nulls)
│
└── tests/                     # 40 Pytest Unit & Integration Tests
    ├── __init__.py
    ├── conftest.py            # Shared data fixtures (clean, dirty, imbalanced)
    ├── test_api.py            # FastAPI endpoints test
    ├── test_correlation.py    # Correlation check tests
    ├── test_data_type_issues.py # Data types check tests
    ├── test_duplicates.py     # Duplicates & split contamination tests
    ├── test_imbalance.py      # Class distribution tests
    ├── test_label_noise.py    # Cross-validated noise detection tests
    ├── test_leakage.py        # Feature leakage detection tests
    ├── test_outliers.py       # Outlier detection tests
    ├── test_profiling.py      # Profiling tests
    ├── test_quality_gate.py   # Quality gate decision boundaries
    └── test_readiness_score.py # Scoring math & penalty tests
```

---

## 🧪 Sample Injected Defects in `sample_customers.csv`

The included customer churn dataset contains realistic anomalies to showcase the checks:
1. **Missing values**: ~9.5% missing records in the `income` column.
2. **Duplicate rows**: 10 duplicate customer rows.
3. **Constant column**: `signup_channel` has only 1 unique value (`web`).
4. **Data Leakage**: `final_account_note` predicts churn alone with ~93% accuracy (simulating post-churn support agent notes).
5. **Label Noise**: 8 deliberately inverted labels caught by cross-validated out-of-fold disagreement.

---

## 📜 License

MIT License. Free to use, adapt, and showcase in personal portfolios.
