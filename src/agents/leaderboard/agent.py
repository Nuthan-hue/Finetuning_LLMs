"""
Leaderboard Monitor Agent
Main agent class for tracking competition leaderboard and performance metrics.
"""
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from ..base import BaseAgent, AgentState
from .fetcher import fetch_leaderboard, get_current_position
from .analysis import generate_recommendation, calculate_performance_trend

logger = logging.getLogger(__name__)


class LeaderboardMonitorAgent(BaseAgent):
    """Agent responsible for monitoring competition leaderboard."""

    def __init__(
        self,
        name: str = "LeaderboardMonitor",
        target_percentile: float = 0.20
    ):
        """
        Initialize the LeaderboardMonitorAgent.

        Args:
            name: Name of the agent
            target_percentile: Target percentile rank to achieve (0.0-1.0)
        """
        super().__init__(name)
        self.target_percentile = target_percentile
        self.leaderboard_history = []

        # Set custom Kaggle config directory (use ~/.kaggle as standard)
        self.kaggle_config_dir = Path.home() / ".kaggle"

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for leaderboard monitoring.

        Args:
            context: Dictionary containing:
                - competition_name: str - Name of Kaggle competition
                - check_interval: int - Seconds between checks (optional)
                - target_percentile: float - Target ranking percentile (optional)

        Returns:
            Dictionary containing:
                - current_rank: Current position on leaderboard
                - total_teams: Total number of teams
                - current_percentile: Current percentile rank
                - meets_target: Whether target is met
                - leaderboard_data: Full leaderboard information
        """
        self.state = AgentState.RUNNING

        try:
            competition_name = context.get("competition_name")
            if not competition_name:
                raise ValueError("competition_name is required")

            # Update target if provided
            if "target_percentile" in context:
                self.target_percentile = context["target_percentile"]

            logger.info(f"Monitoring leaderboard for: {competition_name}")
            logger.info(f"Target percentile: {self.target_percentile * 100}%")

            # Fetch leaderboard
            leaderboard_data = await fetch_leaderboard(
                competition_name,
                self.kaggle_config_dir
            )

            # Get current position
            current_position = await get_current_position(
                competition_name,
                leaderboard_data,
                self.kaggle_config_dir
            )

            # Calculate metrics
            if current_position:
                total_teams = leaderboard_data.get("total_teams", 0)
                current_rank = current_position.get("rank", 0)
                current_percentile = current_rank / total_teams if total_teams > 0 else 1.0
                meets_target = current_percentile <= self.target_percentile

                self.results.update({
                    "current_rank": current_rank,
                    "total_teams": total_teams,
                    "current_percentile": current_percentile,
                    "meets_target": meets_target,
                    "score": current_position.get("score"),
                    "leaderboard_data": leaderboard_data,
                    "timestamp": datetime.now().isoformat()
                })

                # Add to history
                self.leaderboard_history.append(self.results.copy())

                logger.info(f"Current rank: {current_rank}/{total_teams}")
                logger.info(f"Current percentile: {current_percentile * 100:.2f}%")
                logger.info(f"Meets target: {meets_target}")

                # Provide recommendation
                recommendation = await generate_recommendation(
                    current_percentile,
                    self.target_percentile
                )
                self.results["recommendation"] = recommendation

            else:
                logger.warning("Could not find current position on leaderboard")
                self.results["recommendation"] = "make_submission"

            self.state = AgentState.COMPLETED
            return self.results

        except Exception as e:
            error_msg = f"Error monitoring leaderboard: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    def get_performance_trend(self) -> Dict[str, Any]:
        """
        Get performance trend over time.

        Returns:
            Dictionary containing trend information
        """
        return calculate_performance_trend(self.leaderboard_history)

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get full leaderboard history.

        Returns:
            List of historical leaderboard results
        """
        return self.leaderboard_history

    def is_target_met(self) -> bool:
        """
        Check if target percentile is met.

        Returns:
            True if target is met, False otherwise
        """
        if "meets_target" in self.results:
            return self.results["meets_target"]
        return False

    def get_gap_to_target(self) -> Optional[float]:
        """
        Get gap between current and target percentile.

        Returns:
            Gap value (positive means below target) or None if no data
        """
        if "current_percentile" in self.results:
            return self.results["current_percentile"] - self.target_percentile
        return None