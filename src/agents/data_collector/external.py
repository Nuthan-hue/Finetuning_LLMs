"""
External Data Collection
Functions for collecting data from external sources via web scraping.
"""
import logging
from pathlib import Path
from typing import List

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


async def collect_external_data(
    sources: List[str],
    external_dir: Path,
    session: requests.Session
) -> List[str]:
    """
    Collect data from external sources via web scraping.

    Args:
        sources: List of URLs to fetch data from
        external_dir: Directory to save external data
        session: Requests session with configured headers

    Returns:
        List of paths to saved external data files
    """
    logger.info(f"Collecting data from {len(sources)} external sources...")

    external_paths = []

    for i, url in enumerate(sources):
        try:
            logger.info(f"Fetching: {url}")
            response = session.get(url, timeout=30)
            response.raise_for_status()

            # Determine content type and process accordingly
            content_type = response.headers.get('content-type', '')

            if 'text/csv' in content_type or url.endswith('.csv'):
                paths = _save_csv_data(response, external_dir, i)
                external_paths.extend(paths)

            elif 'application/json' in content_type or url.endswith('.json'):
                paths = _save_json_data(response, external_dir, i)
                external_paths.extend(paths)

            elif 'text/html' in content_type:
                paths = scrape_html_tables(response.text, external_dir, i, url)
                external_paths.extend(paths)

            else:
                logger.warning(f"Unsupported content type: {content_type}")

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {str(e)}")
            continue

    logger.info(f"Collected {len(external_paths)} external datasets")
    return external_paths


def _save_csv_data(
    response: requests.Response,
    external_dir: Path,
    index: int
) -> List[str]:
    """
    Save CSV data from HTTP response.

    Args:
        response: HTTP response object
        external_dir: Directory to save data
        index: Index for unique filename

    Returns:
        List containing the saved file path
    """
    filepath = external_dir / f"external_data_{index}.csv"
    with open(filepath, 'wb') as f:
        f.write(response.content)
    logger.info(f"Saved CSV data to {filepath}")
    return [str(filepath)]


def _save_json_data(
    response: requests.Response,
    external_dir: Path,
    index: int
) -> List[str]:
    """
    Save JSON data from HTTP response.

    Args:
        response: HTTP response object
        external_dir: Directory to save data
        index: Index for unique filename

    Returns:
        List containing the saved file path
    """
    filepath = external_dir / f"external_data_{index}.json"
    with open(filepath, 'wb') as f:
        f.write(response.content)
    logger.info(f"Saved JSON data to {filepath}")
    return [str(filepath)]


def scrape_html_tables(
    html_content: str,
    external_dir: Path,
    index: int,
    url: str
) -> List[str]:
    """
    Extract and save tables from HTML content.

    Args:
        html_content: HTML content as string
        external_dir: Directory to save extracted tables
        index: Index for unique filename
        url: Source URL (for logging)

    Returns:
        List of paths to saved table files
    """
    paths = []
    soup = BeautifulSoup(html_content, 'html.parser')
    tables = soup.find_all('table')

    if not tables:
        logger.warning(f"No tables found in HTML from {url}")
        return paths

    for j, table in enumerate(tables):
        try:
            df = pd.read_html(str(table))[0]
            filepath = external_dir / f"external_data_{index}_table_{j}.csv"
            df.to_csv(filepath, index=False)
            paths.append(str(filepath))
            logger.info(f"Extracted table {j} from {url}")
        except Exception as e:
            logger.error(f"Failed to extract table {j}: {str(e)}")

    return paths
