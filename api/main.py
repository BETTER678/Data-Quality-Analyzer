from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import io
import numpy as np

from checks.profiling import profile_dataset
from checks.label_noise import detect_label_noise
from checks.leakage import detect_leakage
from checks.duplicates import detect_duplicates
from checks.imbalance import detect_class_imbalance
from checks.outlier_detection import detect_outliers
from checks.correlation import detect_high_correlation
from checks.data_type_issues import detect_data_type_issues
from report.readiness_score import calculate_readiness_score
from pipeline.quality_gate import apply_quality_gate

app = FastAPI(
    title="Data Quality Analyzer API",
    description="Automated data quality analysis API for machine learning pipelines.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthResponse(BaseModel):
    status: str
    version: str

class CheckInfo(BaseModel):
    name: str
    description: str

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.get("/checks", response_model=List[CheckInfo])
def get_checks():
    return [
        {"name": "profiling", "description": "Basic dataset profiling including missing values, duplicates, and constant columns."},
        {"name": "outliers", "description": "Detects statistical outliers using IQR or Z-score methods."},
        {"name": "correlation", "description": "Identifies highly correlated feature pairs (potential multicollinearity)."},
        {"name": "type_issues", "description": "Finds data type mismatches, mixed types, and casing inconsistencies."},
        {"name": "label_noise", "description": "Detects mislabeled data via cross-validated confident misclassifications."},
        {"name": "leakage", "description": "Detects target leakage from individual features with suspicious solo accuracy."},
        {"name": "imbalance", "description": "Assesses target class distribution and minority percentage."},
        {"name": "duplicates", "description": "Identifies exact duplicate records."},
    ]

def clean_dict(d):
    """Recursively convert numpy and pandas types to standard python types for clean JSON serialization."""
    if isinstance(d, pd.DataFrame):
        return clean_dict(d.replace({np.nan: None}).to_dict(orient="records"))
    elif isinstance(d, pd.Series):
        return clean_dict(d.replace({np.nan: None}).to_dict())
    elif isinstance(d, dict):
        return {str(k): clean_dict(v) for k, v in d.items()}
    elif isinstance(d, (list, tuple, set)):
        return [clean_dict(v) for v in d]
    elif isinstance(d, (np.integer, np.int64, np.int32)):
        return int(d)
    elif isinstance(d, (np.floating, np.float64, np.float32)):
        if np.isnan(d) or np.isinf(d):
            return None
        return float(d)
    elif isinstance(d, np.ndarray):
        return clean_dict(d.tolist())
    elif isinstance(d, np.bool_):
        return bool(d)
    else:
        return d

@app.post("/analyze")
async def analyze_data(
    file: UploadFile = File(...),
    target_col: Optional[str] = Form(None)
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded CSV is empty.")

    if target_col and target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Target column '{target_col}' not found in dataset.")

    # 1. Dataset Info
    dataset_info = {
        "rows": len(df),
        "columns": len(df.columns),
        "columns_list": df.columns.tolist()
    }

    # 2. Checks
    profiling = profile_dataset(df)
    duplicates = detect_duplicates(df)
    outliers = detect_outliers(df)
    correlation = detect_high_correlation(df)
    type_issues = detect_data_type_issues(df)

    label_noise = None
    leakage = None
    imbalance = None

    if target_col:
        label_noise = detect_label_noise(df, target_col)
        leakage = detect_leakage(df, target_col)
        imbalance = detect_class_imbalance(df, target_col)

    # 3. Calculate Readiness Score
    readiness = calculate_readiness_score(
        profile=profiling,
        duplicate_result=duplicates,
        imbalance_result=imbalance,
        label_noise_result=label_noise,
        leakage_result=leakage,
        outlier_result=outliers,
        correlation_result=correlation,
        type_issues_result=type_issues,
    )

    # 4. Apply Quality Gate
    gate = apply_quality_gate(readiness)

    response_data = {
        "dataset_info": dataset_info,
        "profiling": profiling,
        "outliers": outliers,
        "correlation": correlation,
        "type_issues": type_issues,
        "label_noise": label_noise,
        "leakage": leakage,
        "imbalance": imbalance,
        "duplicates": duplicates,
        "readiness_score": readiness,
        "quality_gate": gate,
    }

    return clean_dict(response_data)
