"""
Data Collector
Workflow component for data collection and initial analysis.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import requests

from ..base import BaseAgent, AgentState
from .kaggle_data import download_competition_data
from .analysis import analyze_dataset
from .external import collect_external_data
from .utils import setup_data_directories

logger = logging.getLogger(__name__)


class DataCollector(BaseAgent):
    """Worker responsible for data collection and initial analysis."""

    def __init__(
        self,
        name: str = "DataCollector",
        data_dir: str = "data",
    ):
        """
        Initialize the DataCollector.

        Args:
            name: Name of the worker
            data_dir: Base directory for storing data
        """
        super().__init__(name)
        self.data_dir = Path(data_dir)

        # Set up data directories
        setup_data_directories(self.data_dir)

        # Set custom Kaggle config directory (use ~/.kaggle as standard)
        self.kaggle_config_dir = Path.home() / ".kaggle"
        kaggle_json_path = self.kaggle_config_dir / "kaggle.json"

        # Load Kaggle credentials
        with open(kaggle_json_path, 'r') as f:
            kaggle_config = json.load(f)
            self.kaggle_username = kaggle_config.get("username")
            self.kaggle_key = kaggle_config.get("key")

        # Set up requests session for external data collection
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution method for data collection.

        Args:
            context: Dictionary containing:
                - competition_name: str - Name of the Kaggle competition
                - external_sources: List[str] - URLs for external data (optional)
                - analyze: bool - Whether to perform initial analysis (default: True)

        Returns:
            Dictionary containing:
                - data_path: Path to downloaded data
                - analysis_report: Dict with data statistics
                - external_data_paths: List of paths to external datasets
        """
        print("Starting Data Collector Agent...")
        self.state = AgentState.RUNNING

        try:
            competition_name = context.get("competition_name")
            if not competition_name:
                raise ValueError("competition_name is required in context")

            logger.info(f"Starting data collection for competition: {competition_name}")

            # Download competition data
            data_path = await download_competition_data(
                competition_name,
                self.data_dir,
                self.kaggle_config_dir
            )
            self.results["data_path"] = str(data_path)

            # Analyze data if requested
            if context.get("analyze", True):
                analysis_report = await analyze_dataset(data_path)
                self.results["analysis_report"] = analysis_report

            # Download external data if sources provided
            external_sources = context.get("external_sources", [])
            if external_sources:
                external_dir = self.data_dir / "external"
                external_paths = await collect_external_data(
                    external_sources,
                    external_dir,
                    self.session
                )
                self.results["external_data_paths"] = external_paths

            self.state = AgentState.COMPLETED
            logger.info(f"Data collection completed for {competition_name}")

            return self.results

        except Exception as e:
            error_msg = f"Error during data collection: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary of collected data.

        Returns:
            Analysis report dictionary if available, empty dict otherwise
        """
        if "analysis_report" in self.results:
            return self.results["analysis_report"]
        return {}

    def get_dataset_path(self, dataset_name: str = "train.csv") -> Optional[Path]:
        """
        Get path to a specific dataset file.

        Args:
            dataset_name: Name of the dataset file (default: "train.csv")

        Returns:
            Path to the dataset file if it exists, None otherwise
        """
        if "data_path" in self.results:
            data_path = Path(self.results["data_path"])
            target_file = data_path / dataset_name
            if target_file.exists():
                return target_file
        return None