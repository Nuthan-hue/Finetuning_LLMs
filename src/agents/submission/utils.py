"""
Submission Utilities
Helper functions for submission processing.
"""
import pandas as pd
from typing import Optional
from pathlib import Path


def find_id_column(df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Find ID column in dataframe by trying common names.

    Args:
        df: Input dataframe

    Returns:
        ID column series if found, None otherwise
    """
    id_col_names = ['PassengerId', 'id', 'ID', 'Id', 'test_id']

    for col_name in id_col_names:
        if col_name in df.columns:
            return df[col_name]

    return None


def create_submission_dataframe(
    predictions,
    test_df: pd.DataFrame,
    id_col: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Create submission dataframe with IDs and predictions.

    Args:
        predictions: Model predictions (numpy array or list)
        test_df: Original test dataframe
        id_col: Optional ID column series

    Returns:
        Formatted submission dataframe
    """
    if id_col is None:
        id_col = find_id_column(test_df)

    if id_col is None:
        id_col = range(len(predictions))

    return pd.DataFrame({
        'id': id_col,
        'prediction': predictions
    })