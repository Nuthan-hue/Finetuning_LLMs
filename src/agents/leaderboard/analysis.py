"""
Leaderboard Analysis
Functions for analyzing performance and generating recommendations.
"""
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


async def generate_recommendation(
    current_percentile: float,
    target_percentile: float
) -> str:
    """
    Generate recommendation based on current performance.

    Args:
        current_percentile: Current percentile rank (0.0-1.0)
        target_percentile: Target percentile rank (0.0-1.0)

    Returns:
        Recommendation string indicating suggested action
    """
    gap = current_percentile - target_percentile

    if gap <= 0:
        # Meeting or exceeding target
        if gap < -0.05:
            return "excellent_performance"  # Well above target
        else:
            return "maintain_performance"  # Just above target
    elif gap < 0.05:
        # Close to target
        return "minor_improvement_needed"
    elif gap < 0.15:
        # Moderate gap
        return "retrain_with_tuning"
    else:
        # Large gap
        return "major_improvement_needed"


def calculate_performance_gap(
    current_percentile: float,
    target_percentile: float
) -> float:
    """
    Calculate the gap between current and target percentile.

    Args:
        current_percentile: Current percentile rank (0.0-1.0)
        target_percentile: Target percentile rank (0.0-1.0)

    Returns:
        Gap value (positive means below target, negative means above)
    """
    return current_percentile - target_percentile


def calculate_performance_trend(
    leaderboard_history: list
) -> Dict[str, Any]:
    """
    Calculate performance trend over time.

    Args:
        leaderboard_history: List of historical leaderboard results

    Returns:
        Dictionary containing trend information
    """
    if len(leaderboard_history) < 2:
        return {"trend": "insufficient_data"}

    # Compare latest with previous
    latest = leaderboard_history[-1]
    previous = leaderboard_history[-2]

    rank_change = latest["current_rank"] - previous["current_rank"]
    percentile_change = latest["current_percentile"] - previous["current_percentile"]

    trend = "improving" if rank_change < 0 else "declining" if rank_change > 0 else "stable"

    return {
        "trend": trend,
        "rank_change": rank_change,
        "percentile_change": percentile_change,
        "total_checks": len(leaderboard_history)
    }