"""
Outlier Detection

Checks for:
- Statistical outliers using IQR or Z-score methods on all numeric columns.
"""

import pandas as pd
import numpy as np
from scipy import stats


def detect_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> dict:
    """
    Detect statistical outliers using IQR and Z-score methods on all numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return {
            "supported": False,
            "reason": "No numeric columns found for outlier detection."
        }
        
    outlier_summary = []
    outlier_indices = {}
    all_outlier_rows = set()
    
    for col in numeric_cols:
        series = df[col].dropna()
        if series.empty:
            continue
            
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            col_outliers = series[(series < lower_bound) | (series > upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(series))
            col_outliers = series[z_scores > threshold]
        else:
            raise ValueError(f"Unknown method {method}")
            
        n_outliers = len(col_outliers)
        if n_outliers > 0:
            outlier_pct = round((n_outliers / len(df)) * 100, 2)
            outlier_summary.append({
                'column': col,
                'n_outliers': n_outliers,
                'outlier_pct': outlier_pct,
                'min': col_outliers.min(),
                'max': col_outliers.max(),
                'mean': col_outliers.mean(),
                'median': col_outliers.median()
            })
            idx_list = col_outliers.index.tolist()
            outlier_indices[col] = idx_list
            all_outlier_rows.update(idx_list)
            
    summary_df = pd.DataFrame(outlier_summary)
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=['column', 'n_outliers', 'outlier_pct', 'min', 'max', 'mean', 'median'])
        
    total_outlier_rows = len(all_outlier_rows)
    total_outlier_pct = round((total_outlier_rows / len(df)) * 100, 2) if len(df) > 0 else 0.0
    
    return {
        'supported': True,
        'method': method,
        'outlier_summary': summary_df,
        'total_outlier_rows': total_outlier_rows,
        'total_outlier_pct': total_outlier_pct,
        'outlier_indices': outlier_indices
    }


def summarize_outliers(result: dict) -> str:
    """
    Create a human-readable outlier summary.
    """
    if not result.get("supported"):
        return f"Outlier detection skipped: {result.get('reason')}"
        
    if result["total_outlier_rows"] == 0:
        return f"No outliers detected using {result['method']} method."
        
    return (
        f"Detected {result['total_outlier_rows']} rows with at least one outlier "
        f"({result['total_outlier_pct']}%) using {result['method']} method."
    )
