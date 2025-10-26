# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent system for automating Kaggle competition workflows. The system uses specialized agents (Data Collector, Model Trainer, Submission, Leaderboard Monitor) coordinated by an orchestrator to handle end-to-end competition participation - from data collection to model submission.

## Key Architecture

### Agent System
The codebase implements a multi-agent architecture with:
- **BaseAgent** (src/agents/base.py): Abstract base class defining the agent interface with states (IDLE, RUNNING, COMPLETED, ERROR)
- All agents follow an async execution pattern via the `run(context: Dict) -> Dict` method
- Agents maintain state, error handling, and results through a standardized interface

### Specialized Agents (All Implemented)

#### 1. OrchestratorAgent (src/agents/orchestrator.py)
Central coordinator that manages the entire competition lifecycle:
- Initializes and coordinates all specialized agents
- Implements 4-phase workflow: Data Collection → Training → Submission → Monitoring
- Runs optimization loop until target percentile is met or max iterations reached
- Automatically adjusts training configs based on performance gaps
- Maintains complete workflow history and agent states

#### 2. DataCollectorAgent (src/agents/data_collector.py)
Handles all data acquisition:
- Downloads competition data via Kaggle CLI
- Performs automatic data analysis (shape, dtypes, missing values, target detection)
- Supports external data collection via web scraping
- Extracts tables from HTML pages
- Auto-unzips downloaded datasets

#### 3. ModelTrainerAgent (src/agents/model_trainer.py)
Manages model selection and training:
- Auto-detects task type (tabular/NLP/vision) from data characteristics
- **Tabular models**: LightGBM, XGBoost, PyTorch MLP
  - Automatic categorical encoding and missing value handling
  - Train/val split with early stopping
- **NLP models**: Transformer-based with optional LoRA fine-tuning
  - Auto-detects text columns (avg length > 50 chars)
  - Batch prediction support
- Saves models with best validation scores

#### 4. SubmissionAgent (src/agents/submission.py)
Handles predictions and Kaggle submissions:
- Generates predictions for all model types
- Supports custom submission formatting and column transformations
- Submits via Kaggle CLI with custom messages
- Handles different model formats (.txt for LightGBM, .json for XGBoost, .pt for PyTorch)

#### 5. LeaderboardMonitorAgent (src/agents/leaderboard.py)
Tracks competition performance:
- Fetches leaderboard via Kaggle CLI
- Matches submissions to leaderboard positions
- Calculates current percentile and gap to target
- Generates recommendations: excellent_performance, maintain_performance, minor_improvement_needed, retrain_with_tuning, major_improvement_needed
- Maintains leaderboard history and performance trends

### Supported ML Frameworks
- **Tabular**: LightGBM, XGBoost, PyTorch MLP (all fully implemented)
- **NLP**: Hugging Face Transformers with QLoRA fine-tuning support (implemented)
- **Vision**: CNN architectures (placeholder, not yet implemented)

## Development Commands

### Environment Setup
```bash
# Install core dependencies
pip install -r requirements.txt

# Install with development tools
pip install -e ".[dev]"
```

### Kaggle API Configuration
```bash
# Set up credentials (required for data collection)
mkdir -p ~/.kaggle
# Place kaggle.json in ~/.kaggle/ from https://www.kaggle.com/settings/account
chmod 600 ~/.kaggle/kaggle.json
```

### Running the System
```bash
# Main entry point - interactive menu
python src/main.py

# Direct usage examples:
# 1. Run full automated competition workflow
from agents import OrchestratorAgent
import asyncio

orchestrator = OrchestratorAgent(
    competition_name="titanic",
    target_percentile=0.20,
    max_iterations=5
)
results = asyncio.run(orchestrator.run({}))

# 2. Run individual agents
from agents import DataCollectorAgent
collector = DataCollectorAgent()
data = asyncio.run(collector.run({"competition_name": "titanic"}))

# With Docker/Ollama (pulls gpt-oss model)
docker-compose up
```

### Testing
```bash
# Run tests (when implemented)
pytest tests/

# With coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Format code
black src/

# Sort imports
isort src/

# Linting
flake8 src/

# Type checking
mypy src/
```

## Important Implementation Details

### Agent Communication Pattern
Agents are designed to be coordinated by an orchestrator that manages:
1. Data flow between specialized agents (DataCollector → ModelTrainer → Submission → Leaderboard)
2. Iterative refinement loops based on leaderboard performance
3. State management across the agent lifecycle

### Model Inference Pattern
When extending LocalModel for new model types:
1. Implement model type detection in `_determine_model_type()`
2. Add loading logic in `load_model()`
3. Create corresponding `_generate_*_response()` method
4. Ensure preprocessing matches the model's training format

### Memory and Performance
- Training is designed for systems with 16GB+ RAM (32GB recommended)
- GPU support expected for deep learning models (8GB+ VRAM for transformers)
- Mixed precision training and gradient checkpointing available for large models
- QLoRA used for efficient LLM fine-tuning

## Project Structure Notes

The project follows a modular structure:
- `src/agents/`: Agent implementations (base class exists, others referenced in README)
- `src/models/`: Model-specific implementations organized by type (tabular/nlp/vision)
- `src/data/`: Data processing utilities (preprocessing, augmentation, validation)
- `src/utils/`: Logging, monitoring, and visualization
- `src/config/`: YAML configuration files for models and logging

Note: Many files referenced in the README are planned but not yet implemented in the current codebase.

## Configuration

### Environment Variables
Expected in `.env`:
- `KAGGLE_USERNAME`, `KAGGLE_KEY`: Kaggle API credentials
- `MODEL_TYPE`: lightgbm|xgboost|transformer
- `LOG_LEVEL`: INFO|DEBUG
- `ENABLE_GPU`: true|false

### Docker Setup
The docker-compose.yml runs Ollama locally and pulls the gpt-oss model automatically.