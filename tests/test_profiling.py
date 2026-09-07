import pytest
import pandas as pd
from checks.profiling import profile_dataset, summarize_profile

def test_profile_clean_data(clean_df):
    result = profile_dataset(clean_df)
    assert result["n_rows"] == 100
    assert result["missing_report"].empty
    assert result["n_duplicates"] == 0
    assert len(result["constant_cols"]) == 0

def test_profile_missing_values(dirty_df):
    result = profile_dataset(dirty_df)
    assert not result["missing_report"].empty
    assert "income" in result["missing_report"].index
    # 10 original NaNs + 5 from duplicated rows 0:5 = 15 NaNs
    assert result["missing_report"].loc["income", "missing_count"] == 15

def test_profile_duplicates(dirty_df):
    result = profile_dataset(dirty_df)
    assert result["n_duplicates"] == 5
    assert result["duplicate_pct"] > 0

def test_profile_constant_columns(dirty_df):
    result = profile_dataset(dirty_df)
    assert "constant_col" in result["constant_cols"]

def test_summarize_profile(dirty_df):
    result = profile_dataset(dirty_df)
    summary = summarize_profile(result)
    assert any("missing values" in s for s in summary)
    assert any("duplicate rows found" in s for s in summary)
    assert any("constant column(s) found" in s for s in summary)
