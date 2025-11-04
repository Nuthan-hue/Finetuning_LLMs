"""
Data preprocessing utilities for model training.
Handles data cleaning, feature engineering, and transformations.
"""
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def preprocess_tabular_data(X: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess tabular data for model training.

    Args:
        X: Input DataFrame

    Returns:
        Preprocessed DataFrame
    """
    X_processed = X.copy()

    # Remove ID columns if present (must match submission preprocessing)
    id_columns = ['PassengerId', 'id', 'ID', 'Id']
    for id_col in id_columns:
        if id_col in X_processed.columns:
            X_processed = X_processed.drop(columns=[id_col])

    # Handle categorical variables
    categorical_columns = X_processed.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        # Simple label encoding
        X_processed[col] = pd.Categorical(X_processed[col]).codes

    # Fill missing values
    X_processed = X_processed.fillna(X_processed.mean())

    return X_processed


def scale_features(X_train: np.ndarray, X_val: np.ndarray = None) -> tuple:
    """
    Scale features using StandardScaler.

    Args:
        X_train: Training features
        X_val: Validation features (optional)

    Returns:
        Tuple of (scaler, X_train_scaled, X_val_scaled)
        If X_val is None, returns (scaler, X_train_scaled, None)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    if X_val is not None:
        X_val_scaled = scaler.transform(X_val)
        return scaler, X_train_scaled, X_val_scaled

    return scaler, X_train_scaled, None


def prepare_text_data(df: pd.DataFrame, text_column: str, target_column: str) -> tuple:
    """
    Prepare text data for NLP training.

    Args:
        df: Input DataFrame
        text_column: Name of text column
        target_column: Name of target column

    Returns:
        Tuple of (texts, labels)
    """
    texts = df[text_column].astype(str).tolist()
    labels = df[target_column].tolist()
    return texts, labels


def detect_text_column(df: pd.DataFrame, target_column: str, min_avg_length: int = 50) -> str:
    """
    Auto-detect text column in DataFrame.

    Args:
        df: Input DataFrame
        target_column: Name of target column to exclude
        min_avg_length: Minimum average text length to consider as text column

    Returns:
        Name of detected text column

    Raises:
        ValueError: If no suitable text column is found
    """
    for col in df.columns:
        if col != target_column and df[col].dtype == 'object':
            avg_length = df[col].astype(str).str.len().mean()
            if avg_length > min_avg_length:
                logger.info(f"Detected text column: {col} (avg length: {avg_length:.1f})")
                return col

    raise ValueError("Could not find text column in dataset")


def is_classification_task(y: pd.Series, max_unique_threshold: int = 20) -> bool:
    """
    Determine if task is classification or regression.

    Args:
        y: Target series
        max_unique_threshold: Max unique values to consider as classification

    Returns:
        True if classification, False if regression
    """
    return y.dtype == 'object' or len(y.unique()) < max_unique_threshold