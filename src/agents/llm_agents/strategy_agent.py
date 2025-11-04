"""
Strategy Agent
AI-powered agent that decides optimization strategies based on leaderboard performance.
Replaces hardcoded conditional logic with intelligent reasoning.
"""
import logging
from typing import Dict, Any
from .base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)


class StrategyAgent(BaseLLMAgent):
    """AI agent that intelligently selects optimization strategies."""

    def __init__(self):
        system_prompt = """You are an expert Kaggle competition strategist and machine learning engineer.

Your role is to analyze competition performance and recommend the next optimization strategy.

You have deep knowledge of:
- Machine learning models (LightGBM, XGBoost, Neural Networks, Transformers)
- Hyperparameter tuning strategies
- Feature engineering approaches
- Model ensemble techniques
- Competition-specific optimization tactics

When analyzing performance, consider:
1. Current percentile vs target percentile
2. Performance trends over iterations
3. Which models have been tried
4. The gap magnitude (minor vs major improvements needed)
5. Competition characteristics (tabular, NLP, CV)

Provide actionable, specific recommendations."""

        super().__init__(
            name="StrategyAgent",
            model_name="gemini-1.5-flash",
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

        context = {
            "current_model": current_model,
            "tried_models": ", ".join(tried_models) if tried_models else "None",
            "current_percentile": f"{current_percentile * 100:.2f}%",
            "target_percentile": f"{target_percentile * 100:.2f}%",
            "gap": f"{gap_percentage:.2f}%",
            "iteration": iteration,
            "recommendation": recommendation,
            "competition_type": competition_type,
            "performance_trend": self._analyze_trend(performance_history) if performance_history else "No history"
        }

        prompt = f"""Analyze this Kaggle competition situation and recommend the BEST next optimization strategy.

Current Situation:
- **Iteration**: {iteration}
- **Current Model**: {current_model}
- **Already Tried**: {context['tried_models']}
- **Current Rank**: {context['current_percentile']} percentile
- **Target**: {context['target_percentile']} percentile
- **Gap to Target**: {context['gap']}
- **Competition Type**: {competition_type}
- **Leaderboard Recommendation**: {recommendation}
- **Performance Trend**: {context['performance_trend']}

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

Respond with JSON:
{{
    "action": "switch_model|retrain|tune_aggressive|feature_engineering|ensemble",
    "reasoning": "Detailed explanation of why this is the best strategy",
    "new_model": "model_name if switching, else null",
    "aggressive": true/false,
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
            strategy = await self.reason_json(prompt, context)

            # Ensure required fields
            if "action" not in strategy:
                logger.warning("Strategy missing 'action', using fallback")
                return self._fallback_strategy(current_model, tried_models, gap)

            logger.info(f"AI Strategy: {strategy['action']} - {strategy.get('reasoning', 'No reasoning')}")
            return strategy

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
        elif current_model not in tried_models:
            tried_models.append(current_model)
            action = "switch_model"
            new_model = "xgboost" if current_model == "lightgbm" else "lightgbm"
        else:
            action = "retrain"
            new_model = None

        return {
            "action": action,
            "reasoning": "Fallback strategy due to AI error",
            "new_model": new_model if action == "switch_model" else None,
            "aggressive": gap > 0.15,
            "config_updates": {},
            "expected_improvement": "Unknown",
            "confidence": "low"
        }