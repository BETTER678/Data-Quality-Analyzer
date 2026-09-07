import pytest
import pandas as pd
import numpy as np
from checks.correlation import detect_high_correlation

def test_no_high_correlation(clean_df):
    result = detect_high_correlation(clean_df, threshold=0.95)
    assert result["supported"] is True
    assert result["n_high_pairs"] == 0

def test_high_correlation_detected(clean_df):
    df = clean_df.copy()
    # Create perfectly correlated column
    df['age_copy'] = df['age'] * 2
    
    result = detect_high_correlation(df, threshold=0.95)
    assert result["n_high_pairs"] >= 1
    
    pairs = [tuple(sorted([p[0], p[1]])) for p in result["high_correlation_pairs"]]
    assert tuple(sorted(['age', 'age_copy'])) in pairs
