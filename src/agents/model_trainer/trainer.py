"""
Model Trainer
Workflow component responsible for model selection, training, and optimization.
"""
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd

from ..base import BaseAgent, AgentState

from .detection import TaskType, ModelType, detect_task_type, determine_model_type
from .data_pipeline import DataPipelineExecutor
from .models import (
    train_lightgbm,
    train_xgboost,
    train_pytorch_mlp,
    train_nlp_model
)

logger = logging.getLogger(__name__)


class ModelTrainer(BaseAgent):
    """Worker responsible for model training and optimization."""

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
                - ai_analysis: Dict - Complete AI analysis (REQUIRED)
                - competition_name: str - Competition name
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
        Train a model for tabular data using complete AI-guided pipeline.

        Args:
            data_path: Path to training data
            target_column: Name of target column
            context: Context dictionary with ai_analysis

        Returns:
            Dictionary with training results
        """
        logger.info("Training tabular model with AI-guided preprocessing...")

        # 🤖 GET AI ANALYSIS
        ai_analysis = context.get("ai_analysis", {})
        competition_name = context.get("competition_name", "unknown")

        if not ai_analysis:
            raise RuntimeError(
                "❌ No AI analysis provided! "
                "Pure agentic AI system requires ai_analysis in context."
            )

        # 🤖 EXECUTE COMPLETE AI-GUIDED DATA PIPELINE
        pipeline = DataPipelineExecutor(competition_name)
        X, y = await pipeline.execute(data_path, ai_analysis, target_column)

        config = context.get("config", {})

        # 🤖 AI-RECOMMENDED MODEL SELECTION
        if not self.model_type:
            self.model_type = self._select_model_from_ai(ai_analysis, config)

        logger.info(f"🤖 Using AI-recommended model: {self.model_type.value}")

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

    def _select_model_from_ai(
        self,
        ai_analysis: Dict[str, Any],
        config: Dict[str, Any]
    ) -> ModelType:
        """
        Select model based on AI recommendations.

        Args:
            ai_analysis: AI analysis with recommended_models
            config: User configuration (can override AI)

        Returns:
            ModelType enum
        """
        # Check if user specified a model (overrides AI)
        if "model_type" in config:
            model_str = config["model_type"]
            logger.info(f"🔧 User override: {model_str}")
            if model_str == "xgboost":
                return ModelType.XGBOOST
            elif model_str == "pytorch_mlp":
                return ModelType.PYTORCH_MLP
            elif model_str == "transformer":
                return ModelType.TRANSFORMER
            else:
                return ModelType.LIGHTGBM

        # Use AI recommendation
        recommended_models = ai_analysis.get("recommended_models", [])

        if not recommended_models:
            logger.warning("⚠️  No AI model recommendation - using LightGBM")
            return ModelType.LIGHTGBM

        # Pick first recommended model
        model_name = recommended_models[0].lower()
        logger.info(f"🤖 AI recommends: {model_name}")

        if "xgboost" in model_name:
            return ModelType.XGBOOST
        elif "pytorch" in model_name or "mlp" in model_name or "neural" in model_name:
            return ModelType.PYTORCH_MLP
        elif "lightgbm" in model_name or "lgbm" in model_name:
            return ModelType.LIGHTGBM
        elif "catboost" in model_name:
            # Default to LightGBM if CatBoost not available
            logger.info("   CatBoost not implemented - using LightGBM")
            return ModelType.LIGHTGBM
        else:
            return ModelType.LIGHTGBM