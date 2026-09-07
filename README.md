# Data Quality Analyzer

An automated data quality and readiness assessment tool for machine learning datasets. It checks whether a dataset is safe to train on, highlights hidden issues that can damage model performance, and blocks weak data before it enters a training pipeline.

This project combines a Streamlit dashboard, a FastAPI API, modular quality checks, a scoring engine, and automated reporting into one practical system for evaluating dataset quality.

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Why this exists

Before training a model, the data must be checked for problems that are easy to miss but expensive to fix later.

Typical issues include:

- missing or broken values
- mislabeled examples
- target leakage from a feature that contains the answer indirectly
- duplicate rows or repeated data between train and test splits
- imbalance in the target classes
- outliers that distort training
- high feature correlation and weak schema quality

This project is designed to catch these issues early and give a clear decision: is the dataset ready for training, or should it be reviewed or blocked?

---

## What the project does

The app evaluates a dataset across multiple quality dimensions and produces a single readiness score from 0 to 100.

It checks for:

- missing values and dataset profile issues
- duplicate records and contamination between train/test data
- constant columns with no useful signal
- class imbalance in the target column
- outlier rows using IQR and Z-score logic
- highly correlated features
- numeric values stored as strings
- mixed-type and inconsistent categorical values
- suspicious label noise using cross-validated predictions
- single-feature leakage using model accuracy on individual columns

After the checks run, the system calculates a readiness score and applies a quality gate:

- APPROVED: score >= 80
- REVIEW REQUIRED: score 60 to 79
- BLOCKED: score < 60

---

## Core architecture

```text
Uploaded dataset
      ↓
Quality checks
  ├── Profiling
  ├── Label Noise Detection
  ├── Leakage Detection
  ├── Duplicate Detection
  ├── Class Imbalance Detection
  ├── Outlier Detection
  ├── Correlation Analysis
  └── Type Issue Detection
      ↓
Readiness Score Engine (0-100)
      ↓
Quality Gate
      ↓
Dashboard / API / Report output
```

---

## Features

| Check | Purpose |
|---|---|
| Basic Profiling | Missing values, data types, duplicates, constant columns |
| Label Noise Detection | Flags rows that may have wrong labels |
| Leakage Detection | Finds a single feature that predicts the target too well |
| Duplicate Check | Detects exact duplicate rows and contamination patterns |
| Class Imbalance | Evaluates the target distribution and minority ratio |
| Outlier Detection | Draws attention to abnormal values in numeric columns |
| Correlation Analysis | Detects redundant feature relationships |
| Type Validation | Finds mixed types, numeric strings, and casing inconsistencies |
| Report Generation | Summarizes findings in readable output |

---

## Tech stack

- Python 3.10+
- pandas and numpy
- scikit-learn
- Streamlit
- FastAPI
- Plotly
- pytest

---

## Project structure

```text
Data-Quality-Analyzer/
├── app.py                     # Streamlit dashboard
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image for app/API
├── docker-compose.yml         # Local container configuration
├── README.md                  # Project documentation
├── GETTING_STARTED.md         # Setup guide
├── api/
│   ├── __init__.py
│   └── main.py                # FastAPI endpoints
├── checks/
│   ├── __init__.py
│   ├── profiling.py           # Basic dataset profiling
│   ├── label_noise.py         # Cross-validated label noise detection
│   ├── leakage.py             # Single-feature leakage diagnosis
│   ├── duplicates.py          # Duplicate and contamination checks
│   ├── imbalance.py           # Class imbalance analysis
│   ├── outlier_detection.py   # IQR/Z-score outlier detection
│   ├── correlation.py         # High-correlation checks
│   └── data_type_issues.py    # Type and schema quality checks
├── pipeline/
│   └── quality_gate.py        # Readiness decision logic
├── report/
│   ├── generate_report.py     # Markdown/HTML report generation
│   ├── readiness_score.py     # Score calculation
│   └── visualizations.py      # Plotly visuals for dashboard/reporting
├── sample_data/
│   └── sample_customers.csv   # Example dataset for testing
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_correlation.py
│   ├── test_data_type_issues.py
│   ├── test_duplicates.py
│   ├── test_imbalance.py
│   ├── test_label_noise.py
│   ├── test_leakage.py
│   ├── test_outliers.py
│   ├── test_profiling.py
│   ├── test_quality_gate.py
│   └── test_readiness_score.py
├── .github/
│   └── workflows/
│       └── quality_gate.yml   # Example CI workflow
└── .gitignore
```

---

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/BETTER678/Data-Quality-Analyzer.git
cd Data-Quality-Analyzer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
.\venv\Scripts\activate
```

On macOS/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the dashboard

```bash
streamlit run app.py
```

Then open:

- http://localhost:8501

Upload a CSV file from the sidebar. If your dataset has a target column, select it for ML-specific checks.

---

## Run the API backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:

- http://localhost:8000/docs
- http://localhost:8000/health

Available API endpoints:

- GET /health
- GET /checks
- POST /analyze

Example request:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@sample_data/sample_customers.csv" \
  -F "target_col=churned"
```

---

## Run tests

```bash
pytest -q
```

Or:

```bash
pytest -v
```

---

## Run with Docker

```bash
docker-compose up --build
```

This starts the Streamlit app and API together locally.

---

## Sample dataset

The project includes a sample dataset at [sample_data/sample_customers.csv](sample_data/sample_customers.csv). It contains realistic data issues such as:

- missing values
- duplicates
- constant columns
- imbalance
- noise-like patterns
- leakage-style signal

This makes it useful for testing the full data-quality flow without needing external datasets.

---

## Readiness scoring and quality gate

The scoring engine starts at 100 and subtracts points based on the issues found.

Examples of penalties include:

- missing values
- duplicate records
- constant columns
- label noise
- leakage
- class imbalance
- outliers
- correlation problems

The final result is then mapped to a quality gate:

- APPROVED: score >= 80
- REVIEW REQUIRED: score 60-79
- BLOCKED: score < 60

This gives teams a single decision point before model training begins.

---

## Example output

```text
Dataset Readiness Score: 68/100
Quality Gate: BLOCKED

Issues found:
- Potential data leakage detected
- High potential label noise detected
- Moderate number of duplicate records
- Class imbalance detected

Recommendation: investigate blocked features and re-run analysis before training.
```

---

## Roadmap

Planned improvements include:

- support for larger batch processing
- stronger dataset versioning workflows
- additional export formats for reports
- more advanced leakage and anomaly checks
- deeper CI/CD integration for automated validation

---

## License

This project is licensed under the MIT License.

---

## Summary

This repository provides a practical, end-to-end dataset validation workflow for machine learning projects. It is designed to help teams catch data problems before they reach model training and make that decision clear through a score and a blocking quality gate.
