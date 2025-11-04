"""
Model Trainer Agent
Main agent class responsible for model selection, training, and optimization.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

from ..base import BaseAgent, AgentState

from .detection import TaskType, ModelType, detect_task_type, determine_model_type
from .models import (
    train_lightgbm,
    train_xgboost,
    train_pytorch_mlp,
    train_nlp_model
)

logger = logging.getLogger(__name__)


class ModelTrainerAgent(BaseAgent):
    """Agent responsible for model training and optimization."""

    def __init__(
        self,
        name: str = "ModelTrainer",
        models_dir: str = "models",
        task_type: Optional[TaskType] = None,
        model_type: Optional[ModelType] = None
    ):
        super().__init__(name)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)
        (self.models_dir / "final").mkdir(exist_ok=True)

        self.task_type = task_type
        self.model_type = model_type
        self.model = None
        self.best_score = None

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for model training.

        Args:
            context: Dictionary containing:
                - data_path: str - Path to training data
                - target_column: str - Name of target column
                - task_type: str - Type of task (optional, auto-detected)
                - model_type: str - Specific model to use (optional)
                - config: Dict - Training configuration (optional)

        Returns:
            Dictionary containing:
                - model_path: Path to saved model
                - best_score: Best validation score
                - metrics: Training metrics
        """
        self.state = AgentState.RUNNING

        try:
            data_path = context.get("data_path")
            target_column = context.get("target_column")

            if not data_path or not target_column:
                raise ValueError("data_path and target_column are required")

            logger.info(f"Starting model training with data from: {data_path}")

            # Auto-detect task type if not specified
            if not self.task_type:
                self.task_type = await detect_task_type(data_path, context)

            logger.info(f"Task type: {self.task_type.value}")

            # Select and train model based on task type
            if self.task_type == TaskType.TABULAR:
                results = await self._train_tabular_model(data_path, target_column, context)
            elif self.task_type == TaskType.NLP:
                results = await self._train_nlp_model(data_path, target_column, context)
            elif self.task_type == TaskType.VISION:
                results = await self._train_vision_model(data_path, target_column, context)
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")

            self.results.update(results)
            self.state = AgentState.COMPLETED
            logger.info("Model training completed successfully")

            return self.results

        except Exception as e:
            error_msg = f"Error during model training: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    async def _train_tabular_model(
        self,
        data_path: str,
        target_column: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train a model for tabular data.

        Args:
            data_path: Path to training data
            target_column: Name of target column
            context: Context dictionary with configuration

        Returns:
            Dictionary with training results
        """
        logger.info("Training tabular model...")

        # Load data
        df = pd.read_csv(data_path)

        # Prepare features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        config = context.get("config", {})

        # Determine model type if not specified
        if not self.model_type:
            self.model_type = determine_model_type(config, default=ModelType.LIGHTGBM)

        # Train the appropriate model
        if self.model_type == ModelType.LIGHTGBM:
            results = await train_lightgbm(X, y, config, self.models_dir)
        elif self.model_type == ModelType.XGBOOST:
            results = await train_xgboost(X, y, config, self.models_dir)
        elif self.model_type == ModelType.PYTORCH_MLP:
            results = await train_pytorch_mlp(X, y, config, self.models_dir)
        else:
            raise ValueError(f"Unsupported tabular model: {self.model_type}")

        # Store the model reference
        self.model = results.get("model")
        self.best_score = results.get("best_score")

        return results

    async def _train_nlp_model(
        self,
        data_path: str,
        target_column: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train NLP model with transformer and QLoRA fine-tuning.

        Args:
            data_path: Path to training data
            target_column: Name of target column
            context: Context dictionary with configuration

        Returns:
            Dictionary with training results
        """
        return await train_nlp_model(data_path, target_column, context, self.models_dir)

    async def _train_vision_model(
        self,
        data_path: str,
        target_column: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train computer vision model.

        Args:
            data_path: Path to training data
            target_column: Name of target column
            context: Context dictionary with configuration

        Returns:
            Dictionary with training results
        """
        logger.info("Vision model training not yet fully implemented")
        raise NotImplementedError("Vision model training coming soon")

    def get_model(self):
        """Get the trained model."""
        return self.model