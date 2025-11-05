"""
Data Analysis
Functions for analyzing dataset characteristics.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


async def analyze_dataset(data_path: Path) -> Dict[str, Any]:
    """
    Perform initial data analysis on CSV files in a directory.

    Args:
        data_path: Path to directory containing CSV files

    Returns:
        Dictionary containing analysis results for all datasets
    """
    logger.info("Analyzing dataset...")

    analysis = {
        "files": [],
        "datasets": {}
    }

    try:
        # Find all CSV files
        csv_files = list(data_path.glob("*.csv"))
        analysis["files"] = [f.name for f in csv_files]

        # Analyze each CSV file
        for csv_file in csv_files:
            dataset_info = _analyze_csv_file(csv_file)
            analysis["datasets"][csv_file.name] = dataset_info
            logger.info(
                f"Analyzed {csv_file.name}: "
                f"{dataset_info['shape'][0]} rows, {dataset_info['shape'][1]} columns"
            )

        return analysis

    except Exception as e:
        logger.warning(f"Analysis error: {str(e)}")
        return analysis


def _analyze_csv_file(csv_file: Path) -> Dict[str, Any]:
    """
    Analyze a single CSV file.

    Args:
        csv_file: Path to CSV file

    Returns:
        Dictionary containing dataset information
    """
    df = pd.read_csv(csv_file)

    dataset_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "summary_stats": df.describe().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
    }

    # Identify target column
    target_col = identify_target_column(df)
    if target_col:
        dataset_info["target_column"] = target_col
        dataset_info["target_distribution"] = df[target_col].value_counts().to_dict()

    return dataset_info


def identify_target_column(df: pd.DataFrame) -> Optional[str]:
    """
    Identify the target column in a dataframe.

    NOTE: This function returns None to force AI agent usage.
    Target detection should be handled by DataAnalysisAgent, not hardcoded rules.

    Args:
        df: Pandas DataFrame

    Returns:
        None (AI agent should be used for target detection)
    """
    # Pure AI system - no hardcoded target detection
    # The orchestrator will use DataAnalysisAgent to identify targets
    logger.info("🤖 Target detection requires AI agent - returning None")
    return None


def get_column_types(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Categorize columns by their data types.

    Args:
        df: Pandas DataFrame

    Returns:
        Dictionary mapping type categories to column names
    """
    column_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": []
    }

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            column_types["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types["datetime"].append(col)
        elif pd.api.types.is_object_dtype(dtype):
            # Check if it's likely text or categorical
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:  # High cardinality - likely text
                column_types["text"].append(col)
            else:  # Low cardinality - likely categorical
                column_types["categorical"].append(col)

    return column_types