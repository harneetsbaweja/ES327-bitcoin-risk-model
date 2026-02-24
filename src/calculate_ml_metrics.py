import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np

def calculate_ml_metrics(y_true, y_pred, y_pred_proba, set_name="train"):
    """
    Calculate standard ML classification metrics using sklearn.
    Works for both train and test sets.
    
    Parameters:
    - y_true: true labels
    - y_pred: predicted labels (from predictions_df)
    - y_pred_proba: predicted probabilities (from predictions_df)
    - set_name: "train" or "test" for metric naming
    
    Returns:
    - dict of ML metrics with consistent naming
    """
    ml_metrics = {
        f"{set_name}_accuracy": accuracy_score(y_true, y_pred),
        f"{set_name}_precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        f"{set_name}_recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        f"{set_name}_f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # ROC-AUC only if probabilities available
    if y_pred_proba is not None:
        try:
            # Convert list of arrays back to 2D array for roc_auc_score
            proba_array = np.vstack(y_pred_proba)
            ml_metrics[f"{set_name}_roc_auc"] = roc_auc_score(
                y_true, proba_array, multi_class='ovo', average='weighted'
            )
        except:
            ml_metrics[f"{set_name}_roc_auc"] = None
    
    return ml_metrics