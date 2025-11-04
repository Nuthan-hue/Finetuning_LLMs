"""
Data Preprocessing for Submission
Handles preprocessing of test data before prediction.
"""
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def preprocess_tabular_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess tabular data (same as training).

    Args:
        df: Input dataframe

    Returns:
        Preprocessed dataframe
    """
    df_processed = df.copy()

    # Remove ID column if present
    id_columns = ['PassengerId', 'id', 'ID', 'Id', 'test_id']
    for id_col in id_columns:
        if id_col in df_processed.columns:
            df_processed = df_processed.drop(columns=[id_col])

    # Handle categorical variables
    categorical_columns = df_processed.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        # Simple label encoding
        df_processed[col] = pd.Categorical(df_processed[col]).codes

    # Fill missing values
    df_processed = df_processed.fillna(df_processed.mean())

    return df_processed


def detect_text_column(df: pd.DataFrame, min_avg_length: int = 50) -> str:
    """
    Detect text column in dataframe.

    Args:
        df: Input dataframe
        min_avg_length: Minimum average length to consider as text

    Returns:
        Name of text column

    Raises:
        ValueError: If no text column found
    """
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > min_avg_length:
                logger.info(f"Detected text column: {col} (avg length: {avg_length:.1f})")
                return col

    raise ValueError(f"Could not find text column (min avg length: {min_avg_length})")