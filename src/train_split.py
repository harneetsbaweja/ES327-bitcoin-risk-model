"""
Train/Test Split Module
Handles generic train/test splitting for ML models
"""

from sklearn.model_selection import train_test_split

# ========== CONFIG ==========
train_test_split_ratio = 0.2
random_seed = 42
# ============================


def prepare_train_test_split(feature_data, target_column):
    """
    Generic train/test split - works for ANY ML model.
    
    Args:
        feature_data: DataFrame with features AND target column (already scaled if needed)
        target_column: Name of the target column to predict (e.g., 'Target_1day', 'Price_Up', etc.)
    
    Returns:
        Dictionary with X_train, X_test, y_train, y_test
    """
    # Remove any rows with NaN - index (dates) preserved
    clean_data = feature_data.dropna()
    
    # Separate features from target
    X = clean_data.drop(columns=[target_column], errors='ignore')
    y = clean_data[target_column]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=train_test_split_ratio,
        random_state=random_seed
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
