"""
Data Scaling Module
Handles normalisation and standardisation for dashboard and ML
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_all_features(feature_data):
    """
    Scale ALL features.
    Returns:
        Dictionary with:
        - 'absolute_values': Original DataFrame (with DateTimeIndex preserved)
        - 'normalised_values': DataFrame scaled 0-1 (for dashboard overlay)
        - 'standardised_values': DataFrame with z-scores (for ML and analysis)
        
    Note: All returned DataFrames share the same index (dates), ensuring alignment.
    """
    # Flatten MultiIndex columns if present
    clean_data = feature_data.copy()
    if isinstance(clean_data.columns, pd.MultiIndex):
        clean_data.columns = clean_data.columns.get_level_values(0)
    
    # Remove rows with any NaN - INDEX IS PRESERVED
    clean_data = clean_data.dropna()
    
    # Verify we still have a proper index
    if not isinstance(clean_data.index, pd.DatetimeIndex):
        print(f"Warning: Index is {type(clean_data.index)}, expected DatetimeIndex")
    
    return {
        'absolute_values': clean_data,
        'normalised_values': pd.DataFrame(
            MinMaxScaler().fit_transform(clean_data),
            columns=clean_data.columns,
            index=clean_data.index  # ← EXPLICIT index preservation
        ),
        'standardised_values': pd.DataFrame(
            StandardScaler().fit_transform(clean_data),
            columns=clean_data.columns,
            index=clean_data.index  # ← EXPLICIT index preservation
        )
    }
