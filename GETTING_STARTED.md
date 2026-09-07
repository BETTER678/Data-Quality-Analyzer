# Getting Started with Data Quality Analyzer (v2.0)

Welcome to the **Data Quality Analyzer**! This guide walks you through every layer of the upgraded system: from running the interactive dashboard to testing the REST API and executing automated test suites.

---

## ⚡ Prerequisites

- **Python 3.10+** (Tested up to Python 3.13)
- No API keys required
- No Docker installation required (Docker is optional)

---

## 📦 1. Installation

From the project root directory:

```bash
# Optional: create a virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🖥️ 2. Running the Interactive Dashboard

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Your browser should automatically open `http://localhost:8501`.

### Walkthrough with the Included Sample Data:
1. In the sidebar, click **"Upload CSV Dataset"** and select `sample_data/sample_customers.csv`.
2. Notice the instant overview: **Rows (210)**, **Columns (7)**, **Duplicates (10)**, and missing values distribution.
3. Select `churned` as the **Target Column**.
4. Click on **"🚀 Full Analysis"** in the sidebar navigation and click the big blue **"Run Complete Analysis"** button.
5. Watch all 8 quality checks execute sequentially:
   - **Readiness Score**: Displays with an animated Plotly gauge.
   - **Quality Gate**: Renders the decision (`BLOCKED` or `REVIEW REQUIRED` due to injected leakage & noise).
   - **Quality Radar Chart**: Displays scores across *Completeness*, *Consistency*, *Accuracy*, *Uniqueness*, and *Validity*.
   - **Interactive Charts**: Explore tabs for *Leakage Scores*, *Class Distribution*, *Outlier Box Plots*, and *Correlation Heatmap*.
6. Navigate to **"📊 Report & Export"** in the sidebar to download the assessment as a **Markdown document**, a styled **standalone HTML report**, or a CSV of the flagged label noise rows.

---

## ⚡ 3. Running the REST API

The analyzer can be invoked programmatically by downstream services, automated workflows, or CI/CD pipelines.

Start the FastAPI server:

```bash
uvicorn api.main:app --reload --port 8000
```

### Try the Endpoints:

- **Interactive API Documentation**: Open `http://localhost:8000/docs` in your browser.
- **Health Check**:
  ```bash
  curl http://localhost:8000/health
  ```
- **List All Available Checks**:
  ```bash
  curl http://localhost:8000/checks
  ```
- **Analyze a Dataset**:
  ```bash
  curl -X POST "http://localhost:8000/analyze" \
       -F "file=@sample_data/sample_customers.csv" \
       -F "target_col=churned"
  ```
  Returns a clean, structured JSON payload with all metrics, check outcomes, readiness score, and quality gate ruling.

---

## 🧪 4. Running the Test Suite

Run the full automated test suite using `pytest`:

```bash
pytest tests/ -v
```

All 40 unit and integration tests verify:
- Data profiling edge cases (nulls, zero variance)
- Label noise detection via cross-validation
- Solo predictive leakage detection
- Duplicate records and cross-split contamination
- Target class imbalance
- Statistical outliers (both IQR and Z-Score methods)
- Feature multicollinearity
- Schema and data type mismatches
- Readiness score arithmetic and quality gate boundary rules
- FastAPI endpoints `/health`, `/checks`, and `/analyze`

---

## 🐳 5. Running with Docker (Optional)

If you have Docker Desktop installed, you can containerize the entire stack:

```bash
docker-compose up --build
```
- Dashboard runs on port `8501`
- REST API runs on port `8000`
