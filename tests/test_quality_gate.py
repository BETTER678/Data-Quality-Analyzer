import pytest
import pandas as pd
from pipeline.quality_gate import apply_quality_gate

def test_approved():
    result = apply_quality_gate({"score": 80, "issues": []})
    assert result["decision"] == "APPROVED"
    assert "approved" in result["message"]

def test_review_required():
    result = apply_quality_gate({"score": 75, "issues": ["Issue 1"]})
    assert result["decision"] == "REVIEW REQUIRED"
    assert "moderate quality issues" in result["message"]

def test_blocked():
    result = apply_quality_gate({"score": 50, "issues": ["Issue 1", "Issue 2"]})
    assert result["decision"] == "BLOCKED"
    assert "failed" in result["message"]
