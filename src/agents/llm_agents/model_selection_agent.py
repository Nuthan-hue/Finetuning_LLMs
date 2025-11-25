"""
Model Selection Agent
AI-powered agent that intelligently selects best ML model based on data characteristics and performance history.
"""
import logging
import json
from typing import Dict, Any, List
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class ModelSelectionAgent(BaseLLMAgent):
    """
    AI agent that selects the best model for the task.

    Uses performance history and data characteristics to make intelligent model choices.
    """

    def __init__(self):
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "model_selection_agent.txt"
        system_prompt = prompt_file.read_text() if prompt_file.exists() else self._get_default_prompt()

        super().__init__(
            name="ModelSelectionAgent",
            model_name="llama3.2:7b",  # Lightweight for decision-making
            temperature=0.3,
            system_prompt=system_prompt
        )

    def _get_default_prompt(self) -> str:
        return """You are an expert ML model selection specialist.

Your role: Select the best model for Kaggle competitions based on data characteristics and performance history.

You have access to:
- Data analysis (features, size, modality)
- Performance history (models tried, scores achieved)
- Problem type (classification, regression, etc.)

Available models:
1. **LightGBM** - Fast gradient boosting, good for tabular data
2. **XGBoost** - Powerful gradient boosting, handles missing values well
3. **PyTorch MLP** - Neural network, good for complex patterns
4. **Transformers** - For NLP tasks (BERT, RoBERTa, etc.)

Selection criteria:
- Data size (small: <10K rows, medium: 10K-100K, large: >100K)
- Feature types (numerical, categorical, text)
- Problem complexity
- Previous model performance
- Training time constraints

Return ONLY valid JSON with your selection."""

    async def select_best_model(
        self,
        data_analysis: Dict[str, Any],
        problem_understanding: Dict[str, Any],
        performance_history: List[Dict[str, Any]] = None,
        tried_models: List[str] = None
    ) -> Dict[str, Any]:
        """
        Select the best model for the current task.

        Args:
            data_analysis: Data characteristics from DataAnalysisAgent
            problem_understanding: Problem type from ProblemUnderstandingAgent
            performance_history: List of previous attempts with scores
            tried_models: List of already tried models

        Returns:
            Dictionary with:
                - selected_model: Model name
                - reasoning: Why this model was selected
                - hyperparameters: Suggested hyperparameters
                - training_strategy: How to train
                - confidence: Confidence in selection (0-1)
        """
        logger.info("🤖 Selecting best model based on data characteristics...")

        # Build prompt
        prompt = f"""Analyze the following information and select the best ML model.

## Problem Understanding
{json.dumps(problem_understanding, indent=2)}

## Data Analysis
{json.dumps(data_analysis, indent=2)}

## Performance History
{json.dumps(performance_history or [], indent=2)}

## Already Tried Models
{json.dumps(tried_models or [], indent=2)}

---

Select the best model from: LightGBM, XGBoost, PyTorch MLP, Transformers

Return ONLY a JSON object:
{{
    "selected_model": "lightgbm|xgboost|pytorch_mlp|transformers",
    "reasoning": "Detailed explanation of why this model is best",
    "hyperparameters": {{
        "learning_rate": 0.05,
        "num_iterations": 1000,
        ...
    }},
    "training_strategy": "Cross-validation strategy, early stopping, etc.",
    "confidence": 0.0-1.0,
    "alternatives": ["backup_model1", "backup_model2"]
}}"""

        try:
            response_text = generate_ai_response(self.model, prompt)
            result = self._parse_json_response(response_text)

            logger.info(f"✅ Selected model: {result['selected_model']}")
            logger.info(f"   Confidence: {result.get('confidence', 'N/A')}")
            logger.info(f"   Reasoning: {result['reasoning'][:100]}...")

            return result

        except Exception as e:
            logger.error(f"❌ Model selection failed: {e}")
            # Fallback to LightGBM (most reliable for tabular data)
            return {
                "selected_model": "lightgbm",
                "reasoning": "Fallback to LightGBM due to model selection error",
                "hyperparameters": {
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "num_iterations": 1000
                },
                "training_strategy": "5-fold cross-validation",
                "confidence": 0.5,
                "alternatives": ["xgboost"]
            }

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response into JSON."""
        try:
            # Clean markdown code blocks
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)

            # Validate required fields
            required = ["selected_model", "reasoning", "hyperparameters"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing field: {field}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response: {response_text[:500]}")
            raise