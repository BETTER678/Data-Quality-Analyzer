import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def clean_df():
    """A clean dataset with no issues."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'age': np.random.randint(18, 80, n),
        'income': np.random.normal(50000, 15000, n).round(2),
        'score': np.random.uniform(0, 100, n).round(2),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'target': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    })

@pytest.fixture
def dirty_df():
    """A dataset with injected quality issues."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'id': range(n),
        'age': np.random.randint(18, 80, n).astype(float),
        'income': np.random.normal(50000, 15000, n).round(2),
        'constant_col': ['same'] * n,
        'category': np.random.choice(['A', 'B'], n, p=[0.9, 0.1]),
        'target': np.random.choice([0, 1], n, p=[0.5, 0.5]),
    })
    # Inject missing values
    df.loc[0:9, 'income'] = np.nan
    # Inject outliers
    df.loc[95:99, 'age'] = [200, 250, 300, -50, 999]
    # Inject duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df

@pytest.fixture
def imbalanced_df():
    """Dataset with severe class imbalance."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'feature': np.random.normal(0, 1, n),
        'target': [0]*95 + [1]*5,
    })
