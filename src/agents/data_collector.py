"""
Data Collector Agent
Responsible for downloading competition data, performing initial analysis,
and gathering external datasets.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import subprocess


import pandas as pd
import requests
from bs4 import BeautifulSoup

from .base import BaseAgent, AgentState

logger = logging.getLogger(__name__)


class DataCollectorAgent(BaseAgent):
    """Agent responsible for data collection and initial analysis."""

    def __init__(
        self,
        name: str = "DataCollector",
        data_dir: str = "data",
    ):
        super().__init__(name)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.data_dir / "raw").mkdir(exist_ok=True)
        (self.data_dir / "processed").mkdir(exist_ok=True)
        (self.data_dir / "external").mkdir(exist_ok=True)

        # Set custom Kaggle config directory
        self.kaggle_config_dir = Path("/Volumes/Crucial X9 Pro For Mac/Finetuning_LLMs/.kaggle")
        kaggle_json_path = self.kaggle_config_dir / "kaggle.json"

        with open(kaggle_json_path, 'r') as f:
            kaggle_config = json.load(f)
            self.kaggle_username = kaggle_config.get("username")
            self.kaggle_key = kaggle_config.get("key")

        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        print("Starting Data Collector Agent...")
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
        self.state = AgentState.RUNNING

        try:
            competition_name = context.get("competition_name")
            if not competition_name:
                raise ValueError("competition_name is required in context")

            logger.info(f"Starting data collection for competition: {competition_name}")

            # Download competition data
            data_path = await self._download_competition_data(competition_name)
            self.results["data_path"] = str(data_path)

            # Analyze data if requested
            if context.get("analyze", True):
                analysis_report = await self._analyze_data(data_path)
                self.results["analysis_report"] = analysis_report

            # Download external data if sources provided
            external_sources = context.get("external_sources", [])
            if external_sources:
                external_paths = await self._collect_external_data(external_sources)
                self.results["external_data_paths"] = external_paths

            self.state = AgentState.COMPLETED
            logger.info(f"Data collection completed for {competition_name}")

            return self.results

        except Exception as e:
            error_msg = f"Error during data collection: {str(e)}"
            logger.error(error_msg)
            self.set_error(error_msg)
            raise

    async def _download_competition_data(self, competition_name: str) -> Path:
        """Download competition data using Kaggle API."""
        logger.info(f"Downloading competition data: {competition_name}")

        try:
            # Use Kaggle API via command line

            output_dir = self.data_dir / "raw" / competition_name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Download competition files
            cmd = [
                "kaggle", "competitions", "download",
                "-c", competition_name,
                "-p", str(output_dir)
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

            logger.info(f"Downloaded data to: {output_dir}")

            # Unzip if needed
            zip_files = list(output_dir.glob("*.zip"))
            if zip_files:
                import zipfile
                for zip_file in zip_files:
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(output_dir)
                    logger.info(f"Extracted: {zip_file.name}")

            return output_dir

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Kaggle API error: {e.stderr}")
        except Exception as e:
            raise RuntimeError(f"Failed to download data: {str(e)}")

    async def _analyze_data(self, data_path: Path) -> Dict[str, Any]:
        """Perform initial data analysis."""
        logger.info("Analyzing dataset...")

        analysis = {
            "files": [],
            "datasets": {}
        }

        try:
            # Find all CSV files
            csv_files = list(data_path.glob("*.csv"))
            analysis["files"] = [f.name for f in csv_files]

            # Analyze each CSV file
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)

                dataset_info = {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": df.dtypes.astype(str).to_dict(),
                    "missing_values": df.isnull().sum().to_dict(),
                    "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
                    "summary_stats": df.describe().to_dict(),
                    "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
                }

                # Identify target column (common names)
                potential_targets = ['target', 'label', 'class', 'y', 'outcome']
                target_col = None
                for col in df.columns:
                    if col.lower() in potential_targets:
                        target_col = col
                        break

                if target_col:
                    dataset_info["target_column"] = target_col
                    dataset_info["target_distribution"] = df[target_col].value_counts().to_dict()

                analysis["datasets"][csv_file.name] = dataset_info

                logger.info(f"Analyzed {csv_file.name}: {df.shape[0]} rows, {df.shape[1]} columns")

            return analysis

        except Exception as e:
            logger.warning(f"Analysis error: {str(e)}")
            return analysis

    async def _collect_external_data(self, sources: List[str]) -> List[str]:
        """Collect data from external sources via web scraping."""
        logger.info(f"Collecting data from {len(sources)} external sources...")

        external_paths = []
        external_dir = self.data_dir / "external"

        for i, url in enumerate(sources):
            try:
                logger.info(f"Fetching: {url}")
                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                # Determine content type
                content_type = response.headers.get('content-type', '')

                if 'text/csv' in content_type or url.endswith('.csv'):
                    # Save as CSV
                    filepath = external_dir / f"external_data_{i}.csv"
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    external_paths.append(str(filepath))

                elif 'application/json' in content_type or url.endswith('.json'):
                    # Save as JSON
                    filepath = external_dir / f"external_data_{i}.json"
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    external_paths.append(str(filepath))

                elif 'text/html' in content_type:
                    # Parse HTML and extract tables
                    soup = BeautifulSoup(response.text, 'html.parser')
                    tables = soup.find_all('table')

                    for j, table in enumerate(tables):
                        df = pd.read_html(str(table))[0]
                        filepath = external_dir / f"external_data_{i}_table_{j}.csv"
                        df.to_csv(filepath, index=False)
                        external_paths.append(str(filepath))
                        logger.info(f"Extracted table {j} from {url}")
                else:
                    logger.warning(f"Unsupported content type: {content_type}")

            except Exception as e:
                logger.error(f"Failed to fetch {url}: {str(e)}")
                continue

        logger.info(f"Collected {len(external_paths)} external datasets")
        return external_paths

    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of collected data."""
        if "analysis_report" in self.results:
            return self.results["analysis_report"]
        return {}

    def get_dataset_path(self, dataset_name: str = "train.csv") -> Optional[Path]:
        """Get path to a specific dataset file."""
        if "data_path" in self.results:
            data_path = Path(self.results["data_path"])
            target_file = data_path / dataset_name
            if target_file.exists():
                return target_file
        return None