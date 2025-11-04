"""
Submission Formatting
Handles formatting of predictions into competition-specific submission files.
"""
import logging
import pandas as pd
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


async def auto_detect_format(
    competition_name: str,
    predictions: pd.DataFrame
) -> Dict[str, Any]:
    """
    Auto-detect submission format from sample submission file.

    Args:
        competition_name: Name of the competition
        predictions: Predictions dataframe

    Returns:
        Format specification dictionary
    """
    logger.info("Auto-detecting submission format...")

    # Try to find sample submission in data directory
    data_path = Path("data/raw") / competition_name
    sample_files = list(data_path.glob("*sample*.csv")) + list(data_path.glob("*submission*.csv"))

    if sample_files:
        sample_file = sample_files[0]
        logger.info(f"Found sample submission: {sample_file.name}")

        try:
            sample_df = pd.read_csv(sample_file, nrows=5)
            sample_columns = sample_df.columns.tolist()

            # Detect ID column (usually first column)
            id_col_name = sample_columns[0]

            # Detect prediction column (usually second column or named 'target', 'label', etc.)
            pred_col_name = sample_columns[1] if len(sample_columns) > 1 else "prediction"

            # Determine transformation based on sample values
            transformations = {}
            if pred_col_name in sample_df.columns:
                sample_values = sample_df[pred_col_name].dropna()
                if len(sample_values) > 0:
                    # Check if binary (only 0 and 1)
                    unique_values = sample_values.unique()
                    if len(unique_values) <= 2 and all(v in [0, 1] for v in unique_values):
                        transformations[pred_col_name] = "binary"
                    elif sample_values.dtype in ['int64', 'int32']:
                        transformations[pred_col_name] = "int"
                    elif all(sample_values == sample_values.round()):
                        transformations[pred_col_name] = "round"

            format_spec = {
                "column_mapping": {
                    "id": id_col_name,
                    "prediction": pred_col_name
                },
                "transformations": transformations
            }

            logger.info(f"Detected format - ID: {id_col_name}, Target: {pred_col_name}")
            return format_spec

        except Exception as e:
            logger.warning(f"Could not parse sample submission: {e}")

    # Fallback: use standard format
    logger.warning("Using fallback submission format")
    return {
        "column_mapping": {},
        "transformations": {}
    }


async def format_submission(
    predictions: pd.DataFrame,
    competition_name: str,
    format_spec: Dict[str, Any],
    submissions_dir: Path
) -> Path:
    """
    Format predictions into submission file with auto-detection.

    Args:
        predictions: Predictions dataframe
        competition_name: Name of the competition
        format_spec: Format specification (empty dict for auto-detect)
        submissions_dir: Directory to save submission

    Returns:
        Path to submission file
    """
    logger.info("Formatting submission file...")

    # Auto-detect format if not specified
    if not format_spec:
        format_spec = await auto_detect_format(competition_name, predictions)
        logger.info(f"Auto-detected format: {format_spec}")

    # Apply custom formatting if specified
    if format_spec:
        # Rename columns if specified
        column_mapping = format_spec.get("column_mapping", {})
        if column_mapping:
            predictions = predictions.rename(columns=column_mapping)

        # Apply transformations
        transformations = format_spec.get("transformations", {})
        for col, transform in transformations.items():
            if col in predictions.columns:
                if transform == "round":
                    predictions[col] = predictions[col].round()
                elif transform == "binary":
                    predictions[col] = (predictions[col] > 0.5).astype(int)
                elif transform == "int":
                    predictions[col] = predictions[col].round().astype(int)

    # Save submission file
    submission_filename = f"{competition_name}_submission.csv"
    submission_path = submissions_dir / submission_filename

    predictions.to_csv(submission_path, index=False)
    logger.info(f"Submission saved to: {submission_path}")

    return submission_path