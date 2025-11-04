"""
Kaggle Data Download
Functions for downloading competition data from Kaggle.
"""
import logging
import os
import subprocess
from pathlib import Path

from .utils import extract_zip_files

logger = logging.getLogger(__name__)


async def download_competition_data(
    competition_name: str,
    data_dir: Path,
    kaggle_config_dir: Path
) -> Path:
    """
    Download competition data using Kaggle API.

    Args:
        competition_name: Name of the Kaggle competition
        data_dir: Base data directory
        kaggle_config_dir: Path to Kaggle config directory

    Returns:
        Path to downloaded data directory

    Raises:
        RuntimeError: If download fails
    """
    logger.info(f"Downloading competition data: {competition_name}")

    try:
        # Set up output directory
        output_dir = data_dir / "raw" / competition_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download competition files
        cmd = [
            "kaggle", "competitions", "download",
            "-c", competition_name,
            "-p", str(output_dir)
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

        logger.info(f"Downloaded data to: {output_dir}")

        # Unzip if needed
        extract_zip_files(output_dir)

        return output_dir

    except subprocess.CalledProcessError as e:
        error_msg = f"Kaggle API error (exit code {e.returncode})"
        if e.stderr:
            error_msg += f": {e.stderr}"
        if e.stdout:
            error_msg += f"\nOutput: {e.stdout}"

        # Provide helpful guidance for common errors
        if "403" in str(e.stdout):
            error_msg += f"\n\nHINT: You need to join the competition first!"
            error_msg += f"\n1. Visit: https://www.kaggle.com/competitions/{competition_name}"
            error_msg += f"\n2. Click 'Join Competition' and accept the rules"
            error_msg += f"\n3. Then retry the download"
        elif "404" in str(e.stdout):
            error_msg += f"\n\nHINT: Competition '{competition_name}' not found. Check the competition name/slug."

        raise RuntimeError(error_msg)
    except Exception as e:
        raise RuntimeError(f"Failed to download data: {str(e)}")