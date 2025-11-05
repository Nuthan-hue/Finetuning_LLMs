"""
Data Collector Package
Provides data collection and analysis capabilities for Kaggle competitions.
"""
from .collector import DataCollector
from .kaggle_data import download_competition_data
from .analysis import analyze_dataset, identify_target_column
from .external import collect_external_data, scrape_html_tables
from .utils import setup_data_directories, extract_zip_files

__all__ = [
    # Main worker class
    "DataCollector",

    # Kaggle data functions
    "download_competition_data",

    # Analysis functions
    "analyze_dataset",
    "identify_target_column",

    # External data collection
    "collect_external_data",
    "scrape_html_tables",

    # Utility functions
    "setup_data_directories",
    "extract_zip_files",
]