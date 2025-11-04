"""
Task type and model type detection utilities.
Automatically determines whether data is tabular, NLP, or vision-based.
"""
import logging
from typing import Dict, Any
from pathlib import Path
from enum import Enum
import pandas as pd

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of ML tasks."""
    TABULAR = "tabular"
    NLP = "nlp"
    VISION = "vision"
    UNKNOWN = "unknown"


class ModelType(Enum):
    """Supported model types."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    PYTORCH_MLP = "pytorch_mlp"
    TRANSFORMER = "transformer"
    CNN = "cnn"


async def detect_task_type(data_path: str, context: Dict[str, Any]) -> TaskType:
    """
    Auto-detect the task type based on data characteristics.

    Args:
        data_path: Path to training data
        context: Additional context for detection

    Returns:
        Detected TaskType
    """
    logger.info("Auto-detecting task type...")

    # Check if it's a directory (likely images) or file
    path = Path(data_path)

    if path.is_dir():
        # Check for image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        has_images = any(
            file.suffix.lower() in image_extensions
            for file in path.rglob('*')
        )
        if has_images:
            return TaskType.VISION

    # Load data if it's a file
    if path.suffix == '.csv':
        df = pd.read_csv(data_path)

        # Check for text columns (likely NLP task)
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column contains long text
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 50:  # Arbitrary threshold
                    text_columns.append(col)

        if text_columns:
            logger.info(f"Detected text columns: {text_columns}")
            return TaskType.NLP

        # Default to tabular for structured data
        return TaskType.TABULAR

    return TaskType.UNKNOWN


def determine_model_type(config: Dict[str, Any], default: ModelType = ModelType.LIGHTGBM) -> ModelType:
    """
    Determine model type from configuration.

    Args:
        config: Configuration dictionary
        default: Default model type if not specified

    Returns:
        ModelType enum value
    """
    model_type_str = config.get("model_type", default.value)

    if model_type_str == "xgboost":
        return ModelType.XGBOOST
    elif model_type_str == "pytorch_mlp":
        return ModelType.PYTORCH_MLP
    elif model_type_str == "transformer":
        return ModelType.TRANSFORMER
    elif model_type_str == "cnn":
        return ModelType.CNN
    else:
        return ModelType.LIGHTGBM