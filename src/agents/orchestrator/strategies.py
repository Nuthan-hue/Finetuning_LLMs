"""
Optimization Strategies
Different strategies for improving model performance during iterations.
"""
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


def improve_training_config(
    base_config: Dict[str, Any],
    current_percentile: float,
    target_percentile: float,
    aggressive: bool = False
) -> Dict[str, Any]:
    """
    Improve training configuration based on current performance.

    Args:
        base_config: Base training configuration
        current_percentile: Current leaderboard percentile
        target_percentile: Target percentile to achieve
        aggressive: Whether to use aggressive tuning

    Returns:
        Improved configuration dictionary
    """
    improved_config = base_config.copy()

    gap = current_percentile - target_percentile

    if aggressive or gap > 0.15:
        # Major improvements needed
        logger.info(f"Applying aggressive tuning (gap: {gap:.2%})")
        improved_config.update({
            "num_boost_round": improved_config.get("num_boost_round", 1000) + 500,
            "learning_rate": improved_config.get("learning_rate", 0.05) * 0.5,
            "num_leaves": improved_config.get("num_leaves", 31) + 20,
            "max_depth": improved_config.get("max_depth", 6) + 2,
            "n_estimators": improved_config.get("n_estimators", 1000) + 500
        })
    else:
        # Minor improvements
        logger.info(f"Applying minor tuning (gap: {gap:.2%})")
        improved_config.update({
            "num_boost_round": improved_config.get("num_boost_round", 1000) + 200,
            "learning_rate": improved_config.get("learning_rate", 0.05) * 0.8,
            "n_estimators": improved_config.get("n_estimators", 1000) + 200
        })

    logger.info(f"Updated training config: {improved_config}")
    return improved_config


def should_switch_model(
    current_model: str,
    tried_models: list,
    current_percentile: float,
    target_percentile: float
) -> Tuple[bool, Optional[str]]:
    """
    Determine if we should switch to a different model.

    Args:
        current_model: Currently used model type
        tried_models: List of already tried models
        current_percentile: Current performance percentile
        target_percentile: Target percentile

    Returns:
        Tuple of (should_switch: bool, new_model: Optional[str])
    """
    # Add current model to tried list if not already there
    if current_model not in tried_models:
        tried_models.append(current_model)

    # Try XGBoost if we used LightGBM
    if current_model == "lightgbm" and "xgboost" not in tried_models:
        logger.info("Strategy: Switch from LightGBM to XGBoost")
        return True, "xgboost"

    # Try LightGBM if we used XGBoost
    elif current_model == "xgboost" and "lightgbm" not in tried_models:
        logger.info("Strategy: Switch from XGBoost to LightGBM")
        return True, "lightgbm"

    # Try PyTorch MLP if both gradient boosting models tried
    elif "pytorch_mlp" not in tried_models:
        logger.info("Strategy: Try PyTorch MLP as fallback")
        return True, "pytorch_mlp"

    # No more models to try
    logger.info("Strategy: All models tried, will use hyperparameter tuning")
    return False, None


def select_optimization_strategy(
    recommendation: str,
    current_model: str,
    tried_models: list,
    current_percentile: float,
    target_percentile: float
) -> Dict[str, Any]:
    """
    Select the appropriate optimization strategy based on recommendation.

    Args:
        recommendation: Leaderboard recommendation
        current_model: Currently used model
        tried_models: List of tried models
        current_percentile: Current percentile
        target_percentile: Target percentile

    Returns:
        Strategy dictionary with:
            - action: str ("retrain", "switch_model", "tune_aggressive")
            - new_model: Optional[str]
            - config_updates: Dict
    """
    gap = current_percentile - target_percentile

    # Strategy 1: Minor improvements needed
    if recommendation in ["minor_improvement_needed", "retrain_with_tuning"]:
        return {
            "action": "retrain",
            "new_model": None,
            "config_updates": {},
            "aggressive": False
        }

    # Strategy 2: Major improvements needed
    elif recommendation == "major_improvement_needed":
        return {
            "action": "retrain",
            "new_model": None,
            "config_updates": {},
            "aggressive": True
        }

    # Strategy 3: No clear recommendation - try different approaches
    else:
        # First try switching models
        should_switch, new_model = should_switch_model(
            current_model, tried_models, current_percentile, target_percentile
        )

        if should_switch:
            return {
                "action": "switch_model",
                "new_model": new_model,
                "config_updates": {"model_type": new_model},
                "aggressive": False
            }
        else:
            # All models tried, use aggressive tuning
            return {
                "action": "tune_aggressive",
                "new_model": None,
                "config_updates": {},
                "aggressive": True
            }


def calculate_performance_gap(
    current_percentile: float,
    target_percentile: float
) -> Dict[str, Any]:
    """
    Calculate performance gap and provide analysis.

    Args:
        current_percentile: Current percentile
        target_percentile: Target percentile

    Returns:
        Dictionary with gap analysis
    """
    gap = current_percentile - target_percentile
    gap_percentage = gap * 100

    return {
        "gap": gap,
        "gap_percentage": gap_percentage,
        "meets_target": gap <= 0,
        "severity": (
            "none" if gap <= 0
            else "minor" if gap <= 0.05
            else "moderate" if gap <= 0.10
            else "major" if gap <= 0.20
            else "critical"
        )
    }