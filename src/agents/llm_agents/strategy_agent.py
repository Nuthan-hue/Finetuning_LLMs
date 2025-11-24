"""
Strategy Agent
AI-powered agent that decides optimization strategies based on leaderboard performance.
Replaces hardcoded conditional logic with intelligent reasoning.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class StrategyAgent(BaseLLMAgent):
    """AI agent that intelligently selects optimization strategies."""

    def __init__(self):
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "strategy_agent.txt"
        system_prompt = prompt_file.read_text()

        super().__init__(
            name="StrategyAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.7,
            system_prompt=system_prompt
        )

    async def select_optimization_strategy(
        self,
        recommendation: str,
        current_model: str,
        tried_models: list,
        current_percentile: float,
        target_percentile: float,
        iteration: int,
        competition_type: str = "tabular",
        performance_history: list = None
    ) -> Dict[str, Any]:
        """
        Use AI to select the best optimization strategy.

        Args:
            recommendation: Current recommendation from leaderboard
            current_model: Currently used model
            tried_models: List of already tried models
            current_percentile: Current performance percentile
            target_percentile: Target percentile
            iteration: Current iteration number
            competition_type: Type of competition (tabular, nlp, vision)
            performance_history: Historical performance data

        Returns:
            Strategy dictionary with action and parameters
        """
        gap = current_percentile - target_percentile
        gap_percentage = gap * 100

        tried_models_str = ", ".join(tried_models) if tried_models else "None"
        performance_trend = self._analyze_trend(performance_history) if performance_history else "No history"

        prompt = f"""Analyze this Kaggle competition situation and recommend the BEST next optimization strategy.

Current Situation:
- **Iteration**: {iteration}
- **Current Model**: {current_model}
- **Already Tried**: {tried_models_str}
- **Current Rank**: {current_percentile * 100:.2f}% percentile
- **Target**: {target_percentile * 100:.2f}% percentile
- **Gap to Target**: {gap_percentage:.2f}%
- **Competition Type**: {competition_type}
- **Leaderboard Recommendation**: {recommendation}
- **Performance Trend**: {performance_trend}

Available Actions:
1. **switch_model**: Try a completely different model architecture
2. **retrain**: Retrain current model with improved hyperparameters
3. **tune_aggressive**: Apply aggressive hyperparameter optimization
4. **feature_engineering**: Focus on feature improvements (if applicable)
5. **ensemble**: Create ensemble of best models

Consider:
- Which models work best for {competition_type} competitions?
- Is the gap small (fine-tuning) or large (major changes)?
- What's the performance trend - improving, plateauing, or degrading?
- Have we exhausted model variations?

Phase Dependencies:
- Phase 5 (Planning): Creates execution plan
- Phase 6 (Feature Engineering): Creates new features (only if data_modality != 'tabular')
- Phase 7 (Training): Trains the model
- Phase 8 (Submission): Creates and submits predictions
- Phase 9 (Evaluation): Checks leaderboard
- Phase 10 (Optimization): Decides next steps (this phase)

**Important**: Determine which phases need to rerun based on your action:
- switch_model → [5, 7, 8, 9] (new plan, retrain, resubmit, reevaluate)
- retrain → [7, 8, 9] (retrain with same plan, resubmit, reevaluate)
- tune_aggressive → [5, 7, 8, 9] (new plan with aggressive tuning)
- feature_engineering → [5, 6, 7, 8, 9] (new plan, new features, retrain, resubmit)
- ensemble → [5, 7, 8, 9] (ensemble plan, train ensemble, resubmit)

Respond with JSON:
{{
    "action": "switch_model|retrain|tune_aggressive|feature_engineering|ensemble",
    "reasoning": "Detailed explanation of why this is the best strategy",
    "new_model": "model_name if switching, else null",
    "aggressive": true/false,
    "phases_to_rerun": [5, 7, 8, 9],
    "config_updates": {{
        "model_type": "type if switching",
        "learning_rate": 0.01,
        "num_boost_round": 2000,
        // ... other hyperparameters
    }},
    "expected_improvement": "estimate of expected improvement",
    "confidence": "high|medium|low"
}}"""

        try:
            # Get AI strategy
            logger.info(f"🤖 Requesting strategy from AI (iteration {iteration})...")
            response_text = generate_ai_response(self.model, prompt)

            # Parse JSON response with robust extraction
            cleaned = response_text.strip()

            # Remove markdown code blocks
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Extract JSON from text (handle AI adding explanation after JSON)
            # Find the first { and last } to extract only the JSON part
            import re
            json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = cleaned

            strategy = json.loads(json_str)

            # Ensure required fields
            if "action" not in strategy:
                logger.warning("Strategy missing 'action', using fallback")
                return self._fallback_strategy(current_model, tried_models, gap)

            logger.info(f"✅ AI Strategy: {strategy['action']} - {strategy.get('reasoning', 'No reasoning')[:100]}...")
            return strategy

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            logger.debug(f"Raw AI response (first 500 chars): {response_text[:500]}")
            return self._fallback_strategy(current_model, tried_models, gap)
        except Exception as e:
            logger.error(f"Error in strategy selection: {e}, using fallback")
            return self._fallback_strategy(current_model, tried_models, gap)

    def _analyze_trend(self, performance_history: list) -> str:
        """Analyze performance trend from history."""
        if not performance_history or len(performance_history) < 2:
            return "Insufficient data"

        recent = performance_history[-3:]
        percentiles = [h.get("percentile", 1.0) for h in recent if "percentile" in h]

        if len(percentiles) < 2:
            return "Insufficient data"

        # Check if improving (percentile decreasing is good)
        if percentiles[-1] < percentiles[0]:
            return "Improving"
        elif percentiles[-1] > percentiles[0]:
            return "Degrading"
        else:
            return "Plateauing"

    def _fallback_strategy(
        self,
        current_model: str,
        tried_models: list,
        gap: float
    ) -> Dict[str, Any]:
        """Fallback strategy if AI fails."""
        logger.warning("Using fallback strategy selection")

        # Simple fallback logic
        if gap > 0.15:
            action = "tune_aggressive"
            phases_to_rerun = [5, 7, 8, 9]
        elif current_model not in tried_models:
            tried_models.append(current_model)
            action = "switch_model"
            new_model = "xgboost" if current_model == "lightgbm" else "lightgbm"
            phases_to_rerun = [5, 7, 8, 9]
        else:
            action = "retrain"
            new_model = None
            phases_to_rerun = [7, 8, 9]

        return {
            "action": action,
            "reasoning": "Fallback strategy due to AI error",
            "new_model": new_model if action == "switch_model" else None,
            "aggressive": gap > 0.15,
            "phases_to_rerun": phases_to_rerun,
            "config_updates": {},
            "expected_improvement": "Unknown",
            "confidence": "low"
        }