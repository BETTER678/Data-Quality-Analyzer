import pytest
import pandas as pd
from checks.duplicates import detect_duplicates, check_train_test_contamination

def test_no_duplicates(clean_df):
    result = detect_duplicates(clean_df)
    assert result["n_duplicate_rows"] == 0

def test_with_duplicates(dirty_df):
    result = detect_duplicates(dirty_df)
    assert result["n_duplicate_rows"] == 10 # 5 duplicated rows + original 5 rows = 10 rows flagged as duplicates by keep=False

def test_contamination_no_overlap(clean_df):
    train_df = clean_df.iloc[:50]
    test_df = clean_df.iloc[50:]
    result = check_train_test_contamination(train_df, test_df)
    assert result["supported"] is True
    assert result["n_contaminated_rows"] == 0

def test_contamination_with_overlap(clean_df):
    train_df = clean_df.iloc[:50]
    # Overlap some rows
    test_df = pd.concat([clean_df.iloc[50:], clean_df.iloc[:5]])
    result = check_train_test_contamination(train_df, test_df)
    assert result["supported"] is True
    assert result["n_contaminated_rows"] == 5
