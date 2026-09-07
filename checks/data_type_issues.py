"""
Data Type Issues Detection

Checks for:
- Numeric values stored as strings
- Inconsistent casing in categorical columns
- Mixed types in a single column
"""

import pandas as pd


def detect_data_type_issues(df: pd.DataFrame) -> dict:
    """
    Detect mixed-type and type-mismatch issues.
    """
    if df.empty:
        return {
            "supported": False,
            "reason": "Dataset is empty."
        }

    numeric_as_string = []
    inconsistent_casing = {}
    mixed_types = []
    n_issues = 0
    
    for col in df.columns:
        series = df[col]
        
        # Check for mixed types
        # A simple way to check mixed types is checking the types of the elements
        types = series.dropna().apply(type).unique()
        if len(types) > 1:
            mixed_types.append(col)
            n_issues += 1
            
        # Check object/string columns
        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            # Check numeric as string
            non_null = series.dropna()
            if not non_null.empty:
                # Convert to numeric, coerce errors to NaN
                num_series = pd.to_numeric(non_null, errors='coerce')
                valid_num = num_series.notna().sum()
                
                if valid_num > 0.5 * len(non_null):
                    numeric_as_string.append(col)
                    n_issues += 1
                    
                # Check inconsistent casing
                # Only if they are actually strings
                str_mask = non_null.apply(lambda x: isinstance(x, str))
                str_series = non_null[str_mask]
                
                if not str_series.empty:
                    casing_map = {}
                    for val in str_series.unique():
                        norm_val = val.lower()
                        if norm_val not in casing_map:
                            casing_map[norm_val] = []
                        casing_map[norm_val].append(val)
                        
                    # Filter only those with multiple variants
                    col_casing = {k: v for k, v in casing_map.items() if len(v) > 1}
                    
                    if col_casing:
                        inconsistent_casing[col] = col_casing
                        n_issues += len(col_casing)
                        
    return {
        "supported": True,
        "numeric_as_string": numeric_as_string,
        "inconsistent_casing": inconsistent_casing,
        "mixed_types": mixed_types,
        "n_issues": n_issues
    }


def summarize_data_type_issues(result: dict) -> str:
    """
    Create a human-readable data type issues summary.
    """
    if not result.get("supported", True):
        return f"Data type issue detection skipped: {result.get('reason')}"
        
    if result["n_issues"] == 0:
        return "No data type issues detected."
        
    return f"Detected {result['n_issues']} data type issue(s) across columns."
