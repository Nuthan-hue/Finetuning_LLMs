"""Hyperparameter Optimization Agent - Intelligently searches hyperparameter space"""
import logging
import json
from typing import Dict, Any
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)

class HyperparameterOptAgent(BaseLLMAgent):
    def __init__(self):
        super().__init__(
            name="HyperparameterOptAgent",
            model_name="llama3.2:7b",
            temperature=0.4,
            system_prompt="You optimize ML hyperparameters based on validation performance."
        )

    async def suggest_hyperparameters(
        self,
        model_type: str,
        current_params: Dict[str, Any],
        validation_score: float,
        iteration: int,
        history: list
    ) -> Dict[str, Any]:
        prompt = f"""Suggest next hyperparameters for {model_type}.

Current: {json.dumps(current_params)}
Score: {validation_score}
Iteration: {iteration}
History: {json.dumps(history[-5:] if history else [])}

Return JSON: {{"suggested_params": {{}}, "reasoning": "...", "expected_improvement": 0.02}}"""

        response = generate_ai_response(self.model, prompt)
        return json.loads(response.strip().strip("```json").strip("```"))