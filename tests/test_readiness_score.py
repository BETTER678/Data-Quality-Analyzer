import pytest
import pandas as pd
from report.readiness_score import calculate_readiness_score

def test_perfect_score():
    profile = {
        "missing_report": pd.DataFrame(),
        "constant_cols": [],
    }
    duplicate_result = {"duplicate_percentage": 0.0}
    label_noise_result = {"supported": True, "flagged_pct": 0.0}
    leakage_result = {"supported": True, "n_flagged": 0}
    imbalance_result = {"supported": True, "is_imbalanced": False}
    
    result = calculate_readiness_score(
        profile,
        label_noise_result=label_noise_result,
        leakage_result=leakage_result,
        imbalance_result=imbalance_result,
        duplicate_result=duplicate_result
    )
    assert result["score"] == 100
    assert result["status"] == "PASS"

def test_score_with_issues():
    profile = {
        "missing_report": pd.DataFrame({"missing_pct": [30.0]}),
        "constant_cols": ["col1"],
    }
    # missing pct 30 -> -10, 1 constant col -> -3
    duplicate_result = {"duplicate_percentage": 10.0} # -8
    label_noise_result = {"supported": True, "flagged_pct": 15.0} # -10
    leakage_result = {"supported": True, "n_flagged": 1} # -25
    imbalance_result = {"supported": True, "is_imbalanced": True} # -10
    
    result = calculate_readiness_score(
        profile,
        label_noise_result=label_noise_result,
        leakage_result=leakage_result,
        imbalance_result=imbalance_result,
        duplicate_result=duplicate_result
    )
    
    # 100 - 10 - 3 - 8 - 10 - 25 - 10 = 34
    assert result["score"] == 34
    assert result["status"] == "FAIL"

def test_score_never_below_zero():
    profile = {
        "missing_report": pd.DataFrame({"missing_pct": [100.0]}), # -20
        "constant_cols": ["col1", "col2", "col3", "col4"], # max penalty -10
    }
    duplicate_result = {"duplicate_percentage": 100.0} # -15
    label_noise_result = {"supported": True, "flagged_pct": 100.0} # -20
    leakage_result = {"supported": True, "n_flagged": 5} # -25
    imbalance_result = {"supported": True, "is_imbalanced": True} # -10
    
    result = calculate_readiness_score(
        profile,
        label_noise_result=label_noise_result,
        leakage_result=leakage_result,
        imbalance_result=imbalance_result,
        duplicate_result=duplicate_result
    )
    
    assert result["score"] == 0
    assert result["status"] == "FAIL"

def test_status_pass():
    result = calculate_readiness_score({"missing_report": pd.DataFrame(), "constant_cols": []})
    assert result["status"] == "PASS"

def test_status_warning():
    profile = {
        "missing_report": pd.DataFrame({"missing_pct": [55.0]}), # -20
        "constant_cols": [],
    }
    result = calculate_readiness_score(profile)
    assert result["score"] == 80
    
    profile["constant_cols"] = ["c1"] # -3 => 77
    result = calculate_readiness_score(profile)
    assert result["score"] == 77
    assert result["status"] == "WARNING"

def test_status_fail():
    profile = {
        "missing_report": pd.DataFrame({"missing_pct": [55.0]}), # -20
        "constant_cols": [],
    }
    leakage_result = {"supported": True, "n_flagged": 1} # -25
    result = calculate_readiness_score(profile, leakage_result=leakage_result)
    assert result["score"] == 55
    assert result["status"] == "FAIL"
