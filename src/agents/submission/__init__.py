"""
Submission Package
Handles prediction generation and submission to Kaggle competitions.
"""
from .agent import SubmissionAgent
from .predictions import (
    generate_predictions,
    predict_lightgbm,
    predict_xgboost,
    predict_pytorch_mlp,
    predict_transformer
)
from .formatting import format_submission, auto_detect_format
from .kaggle_api import submit_to_kaggle
from .preprocessing import preprocess_tabular_data, detect_text_column
from .utils import find_id_column, create_submission_dataframe

__all__ = [
    "SubmissionAgent",
    "generate_predictions",
    "predict_lightgbm",
    "predict_xgboost",
    "predict_pytorch_mlp",
    "predict_transformer",
    "format_submission",
    "auto_detect_format",
    "submit_to_kaggle",
    "preprocess_tabular_data",
    "detect_text_column",
    "find_id_column",
    "create_submission_dataframe"
]