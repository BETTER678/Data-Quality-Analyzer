import pytest
import pandas as pd
from checks.data_type_issues import detect_data_type_issues, summarize_data_type_issues

def test_no_type_issues_clean(clean_df):
    result = detect_data_type_issues(clean_df)
    assert result["supported"] is True
    assert result["n_issues"] == 0

def test_numeric_as_string_detected():
    df = pd.DataFrame({
        "num_str": ["10", "20", "30", "40", "hello"],
        "normal": [1, 2, 3, 4, 5]
    })
    result = detect_data_type_issues(df)
    assert "num_str" in result["numeric_as_string"]
    assert result["n_issues"] >= 1

def test_inconsistent_casing_detected():
    df = pd.DataFrame({
        "category": ["Yes", "yes", "YES", "No", "no"]
    })
    result = detect_data_type_issues(df)
    assert "category" in result["inconsistent_casing"]
    assert len(result["inconsistent_casing"]["category"]) >= 1

def test_empty_dataframe():
    df = pd.DataFrame()
    result = detect_data_type_issues(df)
    assert result["supported"] is False
