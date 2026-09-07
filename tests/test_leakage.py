import pytest
import pandas as pd
import numpy as np
from checks.leakage import detect_leakage, summarize_leakage

def test_no_leakage_clean_data(clean_df):
    result = detect_leakage(clean_df, target_col="target")
    assert result["supported"] is True
    assert result["n_flagged"] == 0

def test_leakage_with_leaky_feature(clean_df):
    # Create a feature that is almost identical to the target
    np.random.seed(42)
    leaky_df = clean_df.copy()
    leaky_df["leaky_feat"] = leaky_df["target"].apply(
        lambda x: x if np.random.rand() > 0.1 else 1-x
    )
    result = detect_leakage(leaky_df, target_col="target", suspicion_threshold=0.85)
    assert result["supported"] is True
    assert result["n_flagged"] >= 1
    assert "leaky_feat" in result["flagged_features"]["feature"].values

def test_missing_target(clean_df):
    result = detect_leakage(clean_df, target_col="nonexistent")
    assert result["supported"] is False
