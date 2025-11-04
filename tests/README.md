Let # Test Suite

This directory contains all tests for the Kaggle Competition Multi-Agent System.

## Structure

```
tests/
├── __init__.py                    # Test package initialization
├── conftest.py                    # Pytest fixtures and configuration
├── test_agents/                   # Agent unit tests
│   ├── __init__.py
│   └── test_individual_agents.py  # Individual agent tests
├── test_data/                     # Test data fixtures
├── test_models/                   # Model-specific tests
└── test_orchestrator.py           # End-to-end orchestration tests
```

## Running Tests

### Run Individual Agent Tests

Test all agents sequentially (DataCollector, ModelTrainer, Submission, Leaderboard):

```bash
python3 tests/test_agents/test_individual_agents.py
```

### Run Orchestrator Tests

Test the full end-to-end workflow:

```bash
python3 tests/test_orchestrator.py
```

### Run with Pytest (Recommended)

First install pytest:
```bash
pip install pytest pytest-asyncio
```

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_orchestrator.py -v
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html tests/
```

## Test Coverage

- ✅ **DataCollectorAgent**: Data download, analysis, external sources
- ✅ **ModelTrainerAgent**: LightGBM, XGBoost, PyTorch MLP, NLP transformers
- ✅ **SubmissionAgent**: Prediction generation, formatting, Kaggle submission
- ✅ **LeaderboardMonitorAgent**: Ranking tracking, recommendations
- ✅ **OrchestratorAgent**: Full workflow coordination, optimization loops

## Prerequisites

1. **Kaggle API Credentials**
   ```bash
   # Place kaggle.json in ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **Competition Access**
   - Visit https://www.kaggle.com/competitions/titanic
   - Click "Join Competition" and accept rules

3. **Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Test Configuration

Tests use the following defaults:
- **Competition**: titanic
- **Target Percentile**: Top 20%
- **Max Iterations**: 3
- **Data Directory**: data/
- **Models Directory**: models/
- **Submissions Directory**: submissions/

## Expected Output

### Individual Agent Tests
```
TEST 1: DataCollectorAgent ✓
TEST 2: ModelTrainerAgent ✓
TEST 3: SubmissionAgent ✓
TEST 4: LeaderboardMonitorAgent ✓
ALL TESTS COMPLETED SUCCESSFULLY!
```

### Orchestrator Test
```
PHASE 1: DATA COLLECTION ✓
PHASE 2: INITIAL MODEL TRAINING ✓
PHASE 3: FIRST SUBMISSION ✓
PHASE 4: MONITORING & OPTIMIZATION ✓
ORCHESTRATION COMPLETED!
```

## Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'agents'`
- **Fix**: Tests automatically add `src/` to Python path

**Issue**: `403 Forbidden` when downloading data
- **Fix**: Join the competition at kaggle.com first

**Issue**: `No submissions found` on leaderboard
- **Fix**: Normal - submissions take time to process on Kaggle

**Issue**: Tests take too long
- **Fix**: Reduce `max_iterations` in test_orchestrator.py

## Adding New Tests

1. Create test file in appropriate directory
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures from conftest.py
4. Add docstrings for test functions

Example:
```python
import pytest
from agents.data_collector import DataCollectorAgent

@pytest.mark.asyncio
async def test_data_collector(competition_name, test_data_dir):
    """Test data collection functionality"""
    agent = DataCollectorAgent(data_dir=str(test_data_dir))
    results = await agent.run({"competition_name": competition_name})

    assert results["data_path"]
    assert "analysis_report" in results
```

## CI/CD Integration

To run tests in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest tests/ --junitxml=test-results.xml
  env:
    KAGGLE_USERNAME: ${{ secrets.KAGGLE_USERNAME }}
    KAGGLE_KEY: ${{ secrets.KAGGLE_KEY }}
```