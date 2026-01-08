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
        - 'features_abs': Original DataFrame (with DateTimeIndex preserved)
        - 'features_norm': DataFrame scaled 0-1 (for dashboard overlay)
        - 'features_standardised': DataFrame with z-scores (for ML and analysis)
        
    Note: All returned DataFrames share the same index (dates), ensuring alignment.
    """
    # Flatten MultiIndex columns if present
    feature_data_clean = feature_data.copy()
    if isinstance(feature_data_clean.columns, pd.MultiIndex):
        feature_data_clean.columns = feature_data_clean.columns.get_level_values(0)
    
    # Remove rows with any NaN - INDEX IS PRESERVED
    feature_data_clean = feature_data_clean.dropna()
    
    # Verify we still have a proper index
    if not isinstance(feature_data_clean.index, pd.DatetimeIndex):
        print(f"Warning: Index is {type(feature_data_clean.index)}, expected DatetimeIndex")
    
    return {
        'features_abs': feature_data_clean,
        'features_norm': pd.DataFrame(
            MinMaxScaler().fit_transform(feature_data_clean),
            columns=feature_data_clean.columns,
            index=feature_data_clean.index  # ← EXPLICIT index preservation
        ),
        'features_standardised': pd.DataFrame(
            StandardScaler().fit_transform(feature_data_clean),
            columns=feature_data_clean.columns,
            index=feature_data_clean.index  # ← EXPLICIT index preservation
        )
    }
