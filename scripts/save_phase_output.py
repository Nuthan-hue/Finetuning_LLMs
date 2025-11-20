"""
Centralized utility for saving phase outputs with consistent structure.

Usage:
    from scripts.save_phase_output import save_phase_cache

    save_phase_cache(
        competition_name="titanic",
        phase_number=1,
        phase_name="data_collection",
        data={"data_path": "...", "files": [...]}
    )
"""
import os
import json
from typing import Dict, Any
from pathlib import Path


def save_phase_cache(
    competition_name: str,
    phase_number: int,
    phase_name: str,
    data: Dict[str, Any]
) -> Path:
    """
    Save phase output to cache file.

    Args:
        competition_name: Name of the competition (e.g., "titanic")
        phase_number: Phase number (1-10)
        phase_name: Descriptive phase name (e.g., "data_collection")
        data: Dictionary of data to save

    Returns:
        Path to the saved cache file

    Example:
        save_phase_cache("titanic", 1, "data_collection", {"data_path": "...", "files": [...]})
        # Saves to: data/titanic/phase1_data_collection.json
    """
    # Create cache directory
    cache_dir = Path("data") / competition_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache file path
    cache_file = cache_dir / f"phase{phase_number}_{phase_name}.json"

    # Save to cache
    with open(cache_file, 'w') as f:
        json.dump(data, f, indent=2)

    return cache_file


def load_phase_cache(
    competition_name: str,
    phase_number: int,
    phase_name: str
) -> Dict[str, Any]:
    """
    Load phase output from cache file.

    Args:
        competition_name: Name of the competition (e.g., "titanic")
        phase_number: Phase number (1-10)
        phase_name: Descriptive phase name (e.g., "data_collection")

    Returns:
        Dictionary of cached data, or None if cache doesn't exist

    Example:
        data = load_phase_cache("titanic", 1, "data_collection")
    """
    cache_file = Path("data") / competition_name / f"phase{phase_number}_{phase_name}.json"

    if not cache_file.exists():
        return None

    with open(cache_file, 'r') as f:
        return json.load(f)


def cache_exists(
    competition_name: str,
    phase_number: int,
    phase_name: str
) -> bool:
    """
    Check if phase cache exists.

    Args:
        competition_name: Name of the competition (e.g., "titanic")
        phase_number: Phase number (1-10)
        phase_name: Descriptive phase name (e.g., "data_collection")

    Returns:
        True if cache file exists, False otherwise
    """
    cache_file = Path("data") / competition_name / f"phase{phase_number}_{phase_name}.json"
    return cache_file.exists()
