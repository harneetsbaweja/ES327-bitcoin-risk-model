"""
RandomForest Model Module
Handles RandomForest-specific logic: target creation, training, evaluation
"""

# ========== CONFIG ==========
random_forest_num_trees = 100
random_forest_max_depth = 10
random_forest_random_seed = 42
prediction_time_windows = [1, 7, 30]  # days ahead to predict
# ============================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os


def create_prediction_targets(historical_data, target_price_column=('Close', 'BTC-USD')):
    """
    Create future price targets from historical data.
    
    Args:
        historical_data: DataFrame with historical OHLCV data
        target_price_column: Column to create targets from
    
    Returns:
        DataFrame with Target_1day, Target_7day, Target_30day columns
    """
    targets_df = pd.DataFrame(index=historical_data.index)
    
    for days_ahead in prediction_time_windows:
        targets_df[f'Target_{days_ahead}day'] = historical_data[target_price_column].shift(-days_ahead)
    
    return targets_df


def prepare_features_for_random_forest(features_df):
    """
    Filter features for RandomForest - remove raw OHLCV, keep engineered features only.
    
    Args:
        features_df: DataFrame with all features (including raw OHLCV)
    
    Returns:
        DataFrame with only ML features (RSI, MACD, moving averages, etc.)
    """
    # Remove raw OHLCV columns
    ohlcv_columns = [
        col for col in features_df.columns
        if isinstance(col, tuple) and len(col) == 2 and col[1] == 'BTC-USD'
    ]
    
    ml_features = features_df.drop(ohlcv_columns, axis=1, errors='ignore')
    
    # Flatten MultiIndex if present
    if isinstance(ml_features.columns, pd.MultiIndex):
        ml_features.columns = ml_features.columns.get_level_values(0)
    
    return ml_features


def train_random_forest_model(training_features, training_targets):
    """
    Train a RandomForest regression model to predict Bitcoin prices.
    
    Args:
        training_features: Scaled feature matrix (X_train)
        training_targets: Target prices to predict (y_train)
    
    Returns:
        Trained RandomForest model
    """
    model = RandomForestRegressor(
        n_estimators=random_forest_num_trees,
        max_depth=random_forest_max_depth,
        random_state=random_forest_random_seed,
        n_jobs=-1
    )
    model.fit(training_features, training_targets)
    return model


def evaluate_model_performance(trained_model, test_features, actual_test_prices):
    """
    Evaluate how well the model predicts Bitcoin prices.
    
    Args:
        trained_model: Trained RandomForest model
        test_features: Test features (X_test)
        actual_test_prices: Actual prices (y_test)
    
    Returns:
        Dictionary with predictions, RMSE, and MAE
    """
    predicted_prices = trained_model.predict(test_features)
    
    rmse = np.sqrt(np.mean((predicted_prices - actual_test_prices) ** 2))
    mae = np.mean(np.abs(predicted_prices - actual_test_prices))
    
    return {
        'predictions': predicted_prices,
        'actual_prices': actual_test_prices,
        'rmse': rmse,
        'mae': mae
    }


def save_trained_model(model, prediction_window_days, save_directory='models'):
    """Save trained model to disk."""
    os.makedirs(save_directory, exist_ok=True)
    model_filename = f'{save_directory}/model_{prediction_window_days}day.joblib'
    joblib.dump(model, model_filename)
    print(f"✓ Saved: {model_filename}")


def load_trained_model(prediction_window_days, load_directory='models'):
    """Load trained model from disk."""
    model_filename = f'{load_directory}/model_{prediction_window_days}day.joblib'
    return joblib.load(model_filename)
