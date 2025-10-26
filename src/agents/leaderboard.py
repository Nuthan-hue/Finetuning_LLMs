"""
Leaderboard Monitor Agent
Responsible for tracking competition leaderboard and performance metrics.
"""
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
import subprocess
import re
from datetime import datetime

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class LeaderboardMonitorAgent(BaseAgent):
    """Agent responsible for monitoring competition leaderboard."""

    def __init__(
        self,
        name: str = "LeaderboardMonitor",
        target_percentile: float = 0.20
    ):
        super().__init__(name)
        self.target_percentile = target_percentile
        self.leaderboard_history = []

        # Set custom Kaggle config directory
        self.kaggle_config_dir = Path("/Volumes/Crucial X9 Pro For Mac/Finetuning_LLMs/.kaggle")

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
            leaderboard_data = await self._fetch_leaderboard(competition_name)

            # Get current position
            current_position = await self._get_current_position(
                competition_name,
                leaderboard_data
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
                recommendation = await self._generate_recommendation(
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

    async def _fetch_leaderboard(self, competition_name: str) -> Dict[str, Any]:
        """Fetch leaderboard data from Kaggle."""
        logger.info("Fetching leaderboard data...")

        try:
            cmd = [
                "kaggle", "competitions", "leaderboard",
                competition_name,
                "--show"
            ]

            # Set custom Kaggle config directory in environment
            env = os.environ.copy()
            env["KAGGLE_CONFIG_DIR"] = str(self.kaggle_config_dir)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            # Parse leaderboard output
            leaderboard_data = self._parse_leaderboard(result.stdout)
            return leaderboard_data

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch leaderboard: {e.stderr}")
            return {"error": e.stderr}

    def _parse_leaderboard(self, output: str) -> Dict[str, Any]:
        """Parse Kaggle leaderboard output."""
        lines = output.strip().split('\n')

        leaderboard = {
            "entries": [],
            "total_teams": 0
        }

        # Skip header lines
        data_lines = [line for line in lines if line and not line.startswith('teamId')]

        for line in data_lines:
            # Parse each line (format: teamId,teamName,submissionDate,score)
            parts = line.split(',')
            if len(parts) >= 4:
                entry = {
                    "team_id": parts[0].strip(),
                    "team_name": parts[1].strip(),
                    "submission_date": parts[2].strip(),
                    "score": self._parse_score(parts[3].strip())
                }
                leaderboard["entries"].append(entry)

        leaderboard["total_teams"] = len(leaderboard["entries"])

        return leaderboard

    def _parse_score(self, score_str: str) -> Optional[float]:
        """Parse score string to float."""
        try:
            return float(score_str)
        except (ValueError, AttributeError):
            return None

    async def _get_current_position(
        self,
        competition_name: str,
        leaderboard_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Get current team's position on leaderboard."""
        try:
            # Get submissions to find our team
            cmd = [
                "kaggle", "competitions", "submissions",
                competition_name
            ]

            # Set custom Kaggle config directory in environment
            env = os.environ.copy()
            env["KAGGLE_CONFIG_DIR"] = str(self.kaggle_config_dir)

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )

            # Parse submissions to get latest
            submissions = self._parse_submissions(result.stdout)

            if not submissions:
                logger.warning("No submissions found")
                return None

            # Get latest submission score
            latest_submission = submissions[0]
            latest_score = latest_submission.get("publicScore")

            # Find position in leaderboard
            entries = leaderboard_data.get("entries", [])
            rank = 1

            for entry in entries:
                if entry.get("score") == latest_score:
                    return {
                        "rank": rank,
                        "score": latest_score,
                        "submission": latest_submission
                    }
                rank += 1

            # If exact match not found, estimate position
            logger.warning("Exact position not found, estimating...")
            return {
                "rank": self._estimate_rank(latest_score, entries),
                "score": latest_score,
                "submission": latest_submission,
                "estimated": True
            }

        except Exception as e:
            logger.error(f"Failed to get current position: {str(e)}")
            return None

    def _parse_submissions(self, output: str) -> List[Dict[str, Any]]:
        """Parse Kaggle submissions output."""
        lines = output.strip().split('\n')
        submissions = []

        # Skip header
        data_lines = [line for line in lines if line and not line.startswith('fileName')]

        for line in data_lines:
            parts = line.split(',')
            if len(parts) >= 4:
                submission = {
                    "fileName": parts[0].strip(),
                    "date": parts[1].strip(),
                    "description": parts[2].strip() if len(parts) > 2 else "",
                    "status": parts[3].strip() if len(parts) > 3 else "",
                    "publicScore": self._parse_score(parts[4].strip()) if len(parts) > 4 else None
                }
                submissions.append(submission)

        return submissions

    def _estimate_rank(self, score: float, entries: List[Dict[str, Any]]) -> int:
        """Estimate rank based on score."""
        rank = 1
        for entry in entries:
            entry_score = entry.get("score")
            if entry_score and entry_score > score:
                rank += 1
        return rank

    async def _generate_recommendation(
        self,
        current_percentile: float,
        target_percentile: float
    ) -> str:
        """Generate recommendation based on current performance."""
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

    def get_performance_trend(self) -> Dict[str, Any]:
        """Get performance trend over time."""
        if len(self.leaderboard_history) < 2:
            return {"trend": "insufficient_data"}

        # Compare latest with previous
        latest = self.leaderboard_history[-1]
        previous = self.leaderboard_history[-2]

        rank_change = latest["current_rank"] - previous["current_rank"]
        percentile_change = latest["current_percentile"] - previous["current_percentile"]

        trend = "improving" if rank_change < 0 else "declining" if rank_change > 0 else "stable"

        return {
            "trend": trend,
            "rank_change": rank_change,
            "percentile_change": percentile_change,
            "total_checks": len(self.leaderboard_history)
        }

    def get_history(self) -> List[Dict[str, Any]]:
        """Get full leaderboard history."""
        return self.leaderboard_history

    def is_target_met(self) -> bool:
        """Check if target percentile is met."""
        if "meets_target" in self.results:
            return self.results["meets_target"]
        return False

    def get_gap_to_target(self) -> Optional[float]:
        """Get gap between current and target percentile."""
        if "current_percentile" in self.results:
            return self.results["current_percentile"] - self.target_percentile
        return None
