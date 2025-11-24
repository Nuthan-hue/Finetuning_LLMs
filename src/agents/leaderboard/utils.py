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

    # Skip header and separator lines
    data_lines = []
    for line in lines:
        if line and not line.startswith('teamId') and not line.startswith('---'):
            data_lines.append(line)

    for line in data_lines:
        # Split by whitespace (Kaggle output is space-delimited, not comma-delimited)
        # Format: teamId  teamName  submissionDate  score
        parts = line.split()
        if len(parts) >= 4:
            entry = {
                "team_id": parts[0].strip(),
                "team_name": parts[1].strip(),
                "submission_date": f"{parts[2]} {parts[3]}" if len(parts) > 3 else parts[2],  # Date + time
                "score": parse_score(parts[-1].strip())  # Last column is score
            }
            leaderboard["entries"].append(entry)
            logger.debug(f"Parsed leaderboard entry: Rank {len(leaderboard['entries'])} - {entry['team_name']} - Score: {entry['score']}")

    leaderboard["total_teams"] = len(leaderboard["entries"])
    logger.info(f"Parsed leaderboard with {leaderboard['total_teams']} teams")

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

    # Skip header and separator lines
    data_lines = []
    for line in lines:
        if line and not line.startswith('fileName') and not line.startswith('---'):
            data_lines.append(line)

    for line in data_lines:
        # Split by whitespace (Kaggle output is space-delimited, not comma-delimited)
        parts = line.split()
        if len(parts) >= 5:  # Need at least: fileName, date, time, status, publicScore
            # Kaggle format: fileName  date  time  description  status  publicScore
            # Handle case where description might be missing or multiple words
            submission = {
                "fileName": parts[0].strip(),
                "date": f"{parts[1]} {parts[2]}",  # Combine date and time
                "publicScore": parse_score(parts[-1].strip()) if parts[-1] else None,  # Last column is publicScore
                "status": parts[-2].strip() if len(parts) > 4 else "",  # Second to last is status
                "description": " ".join(parts[3:-2]) if len(parts) > 5 else "",  # Everything in between
            }
            submissions.append(submission)
            logger.debug(f"Parsed submission: {submission['fileName']} - Score: {submission['publicScore']}")

    logger.info(f"Parsed {len(submissions)} submissions")
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