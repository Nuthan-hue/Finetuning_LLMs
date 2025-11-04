"""
Kaggle API Integration
Handles submission to Kaggle competitions.
"""
import os
import logging
import subprocess
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


async def submit_to_kaggle(
    submission_path: Path,
    competition_name: str,
    message: str,
    kaggle_config_dir: Path
) -> Dict[str, Any]:
    """
    Submit to Kaggle competition.

    Args:
        submission_path: Path to submission file
        competition_name: Name of the competition
        message: Submission message
        kaggle_config_dir: Path to Kaggle config directory

    Returns:
        Dictionary with submission status and details
    """
    logger.info(f"Submitting to Kaggle competition: {competition_name}")

    try:
        cmd = [
            "kaggle", "competitions", "submit",
            "-c", competition_name,
            "-f", str(submission_path),
            "-m", message
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

        logger.info("Submission successful!")
        logger.info(result.stdout)

        return {
            "submission_status": "success",
            "submission_message": result.stdout
        }

    except subprocess.CalledProcessError as e:
        error_msg = f"Kaggle submission failed: {e.stderr}"
        logger.error(error_msg)
        return {
            "submission_status": "failed",
            "submission_error": error_msg
        }