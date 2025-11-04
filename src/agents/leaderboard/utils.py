"""
Leaderboard Utilities
Helper functions for parsing and processing leaderboard data.
"""
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def parse_leaderboard(output: str) -> Dict[str, Any]:
    """
    Parse Kaggle leaderboard output.

    Args:
        output: Raw output from Kaggle leaderboard command

    Returns:
        Dictionary containing parsed leaderboard entries and total teams
    """
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
                "score": parse_score(parts[3].strip())
            }
            leaderboard["entries"].append(entry)

    leaderboard["total_teams"] = len(leaderboard["entries"])

    return leaderboard


def parse_submissions(output: str) -> List[Dict[str, Any]]:
    """
    Parse Kaggle submissions output.

    Args:
        output: Raw output from Kaggle submissions command

    Returns:
        List of parsed submission dictionaries
    """
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
                "publicScore": parse_score(parts[4].strip()) if len(parts) > 4 else None
            }
            submissions.append(submission)

    return submissions


def parse_score(score_str: str) -> Optional[float]:
    """
    Parse score string to float.

    Args:
        score_str: String representation of score

    Returns:
        Float value of score or None if parsing fails
    """
    try:
        return float(score_str)
    except (ValueError, AttributeError):
        return None


def estimate_rank(score: float, entries: List[Dict[str, Any]]) -> int:
    """
    Estimate rank based on score compared to leaderboard entries.

    Args:
        score: The score to estimate rank for
        entries: List of leaderboard entries

    Returns:
        Estimated rank position
    """
    rank = 1
    for entry in entries:
        entry_score = entry.get("score")
        if entry_score and entry_score > score:
            rank += 1
    return rank