"""Ensemble Strategy Agent - Decides model ensembling strategies"""
import logging
import json
from typing import Dict, Any, List
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)

class EnsembleStrategyAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__(
            name="EnsembleStrategyAgent",
            model_name="llama3.3:70b",  # Complex strategy needs best model
            temperature=0.5,
            system_prompt="You are an ensemble strategy expert for ML competitions."
        )

    async def design_ensemble(
        self,
        trained_models: List[Dict[str, Any]],
        validation_scores: Dict[str, float],
        problem_type: str
    ) -> Dict[str, Any]:
        prompt = f"""Design ensemble strategy.

Models: {json.dumps(trained_models, indent=2)}
Scores: {json.dumps(validation_scores)}
Problem: {problem_type}

Return JSON:
{{
    "should_ensemble": true/false,
    "ensemble_method": "voting|stacking|blending",
    "models_to_use": ["model1", "model2"],
    "weights": {{"model1": 0.6, "model2": 0.4}},
    "reasoning": "...",
    "expected_score": 0.95
}}"""

        response = generate_ai_response(self.model, prompt)
        return json.loads(response.strip().strip("```json").strip("```"))