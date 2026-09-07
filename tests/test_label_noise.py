import pytest
from checks.label_noise import detect_label_noise, summarize_label_noise

def test_label_noise_clean(clean_df):
    df = clean_df.copy()
    # Distinct separable pattern ensures no model confusion
    df["target"] = (df["age"] > 45).astype(int)
    result = detect_label_noise(df, target_col="target")
    assert result["supported"] is True
    assert result["n_flagged"] == 0

def test_unsupported_target(clean_df):
    result = detect_label_noise(clean_df, target_col="age")
    # Age has too many unique values for classification
    assert result["supported"] is False
    assert "doesn't qualify" in result["reason"]

def test_missing_target_col(clean_df):
    result = detect_label_noise(clean_df, target_col="nonexistent")
    assert result["supported"] is False
    assert "not found" in result["reason"]
