"""
Leaderboard Fetcher
Functions for fetching leaderboard and submission data from Kaggle.
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

from .utils import parse_leaderboard, parse_submissions, estimate_rank

logger = logging.getLogger(__name__)


async def fetch_leaderboard(
    competition_name: str,
    kaggle_config_dir: Path
) -> Dict[str, Any]:
    """
    Fetch leaderboard data from Kaggle.

    Args:
        competition_name: Name of the competition
        kaggle_config_dir: Path to Kaggle config directory

    Returns:
        Dictionary containing parsed leaderboard data
    """
    logger.info("Fetching leaderboard data...")

    try:
        cmd = [
            "kaggle", "competitions", "leaderboard",
            competition_name,
            "--show"
        ]

        # Set custom Kaggle config directory in environment
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        # Parse leaderboard output
        leaderboard_data = parse_leaderboard(result.stdout)
        return leaderboard_data

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to fetch leaderboard: {e.stderr}")
        return {"error": e.stderr}


async def get_current_position(
    competition_name: str,
    leaderboard_data: Dict[str, Any],
    kaggle_config_dir: Path
) -> Optional[Dict[str, Any]]:
    """
    Get current team's position on leaderboard.

    Args:
        competition_name: Name of the competition
        leaderboard_data: Parsed leaderboard data
        kaggle_config_dir: Path to Kaggle config directory

    Returns:
        Dictionary containing rank, score, and submission info, or None if not found
    """
    try:
        # Get submissions to find our team
        cmd = [
            "kaggle", "competitions", "submissions",
            competition_name
        ]

        # Set custom Kaggle config directory in environment
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(kaggle_config_dir)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            env=env
        )

        # Parse submissions to get latest
        submissions = parse_submissions(result.stdout)

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
            "rank": estimate_rank(latest_score, entries),
            "score": latest_score,
            "submission": latest_submission,
            "estimated": True
        }

    except Exception as e:
        logger.error(f"Failed to get current position: {str(e)}")
        return None