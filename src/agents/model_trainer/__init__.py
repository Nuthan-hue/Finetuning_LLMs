"""
Model Trainer Package
Provides model training capabilities for tabular, NLP, and vision tasks.
"""
from .trainer import ModelTrainer
from .detection import TaskType, ModelType, detect_task_type, determine_model_type
from .models import (
    train_lightgbm,
    train_xgboost,
    train_pytorch_mlp,
    train_nlp_model
)
from .preprocessing import (
    preprocess_tabular_data,
    scale_features,
    prepare_text_data,
    detect_text_column,
    is_classification_task
)

__all__ = [
    # Main worker class
    "ModelTrainer",

    # Detection utilities
    "TaskType",
    "ModelType",
    "detect_task_type",
    "determine_model_type",

    # Model training functions
    "train_lightgbm",
    "train_xgboost",
    "train_pytorch_mlp",
    "train_nlp_model",

    # Preprocessing utilities
    "preprocess_tabular_data",
    "scale_features",
    "prepare_text_data",
    "detect_text_column",
    "is_classification_task",
]