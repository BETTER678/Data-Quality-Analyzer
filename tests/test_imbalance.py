import pytest
from checks.imbalance import detect_class_imbalance, summarize_class_imbalance

def test_balanced_classes(clean_df):
    result = detect_class_imbalance(clean_df, target_col="target")
    assert result["supported"] is True
    assert result["is_imbalanced"] == False

def test_imbalanced_classes(imbalanced_df):
    result = detect_class_imbalance(imbalanced_df, target_col="target")
    assert result["supported"] is True
    assert result["is_imbalanced"] == True
    assert result["minority_percentage"] == 5.0

def test_missing_target(clean_df):
    result = detect_class_imbalance(clean_df, target_col="nonexistent")
    assert result["supported"] is False
