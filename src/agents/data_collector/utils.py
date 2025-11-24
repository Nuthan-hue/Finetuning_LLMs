"""
Data Collector Utilities
Helper functions for data collection operations.
"""
import logging
import zipfile
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def setup_data_directories(data_dir: Path) -> None:
    """
    Create base data directory.
    Individual competition directories (raw/, processed/, featured/, metadata/)
    are created on-demand by each phase.

    Args:
        data_dir: Base data directory path
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Data directory set up at: {data_dir}")


def extract_zip_files(directory: Path) -> List[str]:
    """
    Extract all ZIP files in a directory.

    Args:
        directory: Directory containing ZIP files

    Returns:
        List of extracted file paths
    """
    extracted_files = []
    zip_files = list(directory.glob("*.zip"))

    if not zip_files:
        return extracted_files

    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(directory)
                extracted_files.extend(zip_ref.namelist())
                logger.info(f"Extracted: {zip_file.name}")
        except Exception as e:
            logger.error(f"Failed to extract {zip_file.name}: {str(e)}")

    return extracted_files