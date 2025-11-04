"""
Pytest configuration and shared fixtures
"""
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def competition_name():
    """Default competition name for tests"""
    return "titanic"


@pytest.fixture
def test_data_dir(tmp_path):
    """Temporary directory for test data"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def test_models_dir(tmp_path):
    """Temporary directory for test models"""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    return models_dir


@pytest.fixture
def test_submissions_dir(tmp_path):
    """Temporary directory for test submissions"""
    submissions_dir = tmp_path / "submissions"
    submissions_dir.mkdir()
    return submissions_dir