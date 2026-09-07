"""
Correlation Analysis

Checks for:
- Highly correlated feature pairs.
"""

import pandas as pd
import numpy as np


def detect_high_correlation(df: pd.DataFrame, threshold: float = 0.95) -> dict:
    """
    Detect highly correlated feature pairs.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return {
            "supported": False,
            "reason": "Not enough numeric columns for correlation analysis."
        }
        
    corr_matrix = df[numeric_cols].corr(method='pearson')
    
    high_corr_pairs = []
    
    # Iterate over upper triangle of correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_val = corr_matrix.iloc[i, j]
            
            if pd.notna(corr_val) and abs(corr_val) >= threshold:
                high_corr_pairs.append((col1, col2, float(corr_val)))
                
    return {
        "supported": True,
        "correlation_matrix": corr_matrix,
        "high_correlation_pairs": high_corr_pairs,
        "n_high_pairs": len(high_corr_pairs),
        "threshold": threshold
    }


def summarize_correlation(result: dict) -> str:
    """
    Create a human-readable correlation summary.
    """
    if not result.get("supported"):
        return f"Correlation analysis skipped: {result.get('reason')}"
        
    if result["n_high_pairs"] == 0:
        return f"No highly correlated features detected (threshold: {result['threshold']})."
        
    return (
        f"Detected {result['n_high_pairs']} highly correlated feature pair(s) "
        f"(|r| >= {result['threshold']})."
    )
