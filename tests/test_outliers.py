import pytest
import pandas as pd
from checks.outlier_detection import detect_outliers

def test_no_outliers_clean(clean_df):
    result = detect_outliers(clean_df, method='iqr', threshold=1.5)
    assert result["supported"] is True
    assert result["total_outlier_rows"] == 0

def test_outliers_detected(dirty_df):
    result = detect_outliers(dirty_df, method='iqr', threshold=1.5)
    assert result["total_outlier_rows"] > 0
    age_outliers = result["outlier_indices"].get("age", [])
    assert 95 in age_outliers or 99 in age_outliers

def test_iqr_method(dirty_df):
    result = detect_outliers(dirty_df, method='iqr', threshold=1.5)
    assert result["method"] == 'iqr'
    assert result["total_outlier_rows"] > 0

def test_zscore_method(dirty_df):
    result = detect_outliers(dirty_df, method='zscore', threshold=3.0)
    assert result["method"] == 'zscore'
    assert result["total_outlier_rows"] > 0
