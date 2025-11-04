# 📁 Agent Refactoring Guide

## Goal
Split large agent files into modular folder structures for better maintainability.

---

## Current Structure (❌ Large Files)

```
src/agents/
├── base.py                    (50 lines)
├── data_collector.py          (281 lines) ← Too large!
├── model_trainer.py           (608 lines) ← Too large!
├── submission.py              (521 lines) ← Too large!
├── leaderboard.py             (336 lines) ← Large
└── orchestrator.py            (505 lines) ← Too large!
```

**Total:** 2,301 lines in 6 files

---

## Target Structure (✅ Modular)

```
src/agents/
├── base/
│   ├── __init__.py            # Export BaseAgent, AgentState
│   └── agent.py               # BaseAgent class (50 lines)
│
├── data_collector/
│   ├── __init__.py            # Export DataCollectorAgent
│   ├── collector.py           # Main agent class (100 lines)
│   └── analyzers.py           # Data analysis functions (180 lines)
│
├── model_trainer/
│   ├── __init__.py            # Export ModelTrainerAgent
│   ├── trainer.py             # Main orchestration (150 lines)
│   ├── tabular.py             # LightGBM, XGBoost, MLP (250 lines)
│   ├── nlp.py                 # Transformers with LoRA (180 lines)
│   └── vision.py              # CNN models (30 lines)
│
├── submission/
│   ├── __init__.py            # Export SubmissionAgent
│   ├── submitter.py           # Main agent class (150 lines)
│   ├── predictors.py          # Generate predictions (250 lines)
│   └── formatters.py          # Format submissions (120 lines)
│
├── leaderboard/
│   ├── __init__.py            # Export LeaderboardMonitorAgent
│   ├── monitor.py             # Main agent class (180 lines)
│   └── analyzer.py            # Recommendations (150 lines)
│
└── orchestrator/
    ├── __init__.py            # Export OrchestratorAgent
    ├── orchestrator.py        # Main workflow (300 lines)
    └── strategies.py          # Optimization strategies (200 lines)
```

**Total:** 2,340 lines in 20 files (same code, better organized!)

---

## Step-by-Step Refactoring

### Step 1: Refactor BaseAgent ✅

**Create:** `src/agents/base/agent.py`

```python
"""
Base Agent Module
Provides the abstract base class for all agents.
"""
from enum import Enum
from typing import Any, Dict

class AgentState(Enum):
    """Possible states for an agent."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class BaseAgent:
    """Abstract base class for all agents."""
    # ... rest of BaseAgent code from base.py
```

**Create:** `src/agents/base/__init__.py`

```python
"""Base Agent Module"""
from .agent import BaseAgent, AgentState

__all__ = ['BaseAgent', 'AgentState']
```

**Delete:** `src/agents/base.py` (after migration)

---

### Step 2: Refactor DataCollectorAgent

**Create:** `src/agents/data_collector/collector.py`

```python
"""
Data Collector Agent
Main agent class for data collection.
"""
import logging
from pathlib import Path
from typing import Dict, Any, List
from agents.base import BaseAgent, AgentState
from .analyzers import analyze_dataset, detect_target_column

logger = logging.getLogger(__name__)

class DataCollectorAgent(BaseAgent):
    """Agent responsible for collecting and analyzing competition data."""

    def __init__(self, name: str = "DataCollector", data_dir: str = "data"):
        super().__init__(name)
        # ... initialization code

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method."""
        # ... main logic (keep it under 100 lines)
```

**Create:** `src/agents/data_collector/analyzers.py`

```python
"""
Data Analysis Utilities
Functions for analyzing datasets and detecting patterns.
"""
import pandas as pd
from typing import Dict, Any

def analyze_dataset(file_path: str) -> Dict[str, Any]:
    """Analyze a single dataset file."""
    df = pd.read_csv(file_path)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
    }

def detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect the target column."""
    # ... detection logic
    pass

def extract_table_from_html(html_content: str) -> pd.DataFrame:
    """Extract table from HTML."""
    # ... extraction logic
    pass
```

**Create:** `src/agents/data_collector/__init__.py`

```python
"""Data Collector Module"""
from .collector import DataCollectorAgent

__all__ = ['DataCollectorAgent']
```

---

### Step 3: Refactor ModelTrainerAgent

**Create:** `src/agents/model_trainer/trainer.py`

```python
"""
Model Trainer Agent
Main orchestration for model training.
"""
from agents.base import BaseAgent, AgentState
from .tabular import train_lightgbm, train_xgboost, train_pytorch_mlp
from .nlp import train_transformer
from .vision import train_cnn

class ModelTrainerAgent(BaseAgent):
    """Agent responsible for model training."""

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution - delegates to specific trainers."""

        # Detect task type
        task_type = await self._detect_task_type(data_path, context)

        # Delegate to appropriate trainer
        if task_type == TaskType.TABULAR:
            return await self._train_tabular(data_path, target_column, context)
        elif task_type == TaskType.NLP:
            return await train_transformer(data_path, target_column, context)
        # ...
```

**Create:** `src/agents/model_trainer/tabular.py`

```python
"""
Tabular Model Training
LightGBM, XGBoost, PyTorch MLP implementations.
"""
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn

async def train_lightgbm(X, y, config):
    """Train LightGBM model."""
    # ... 80 lines of LightGBM training code

async def train_xgboost(X, y, config):
    """Train XGBoost model."""
    # ... 80 lines of XGBoost training code

async def train_pytorch_mlp(X, y, config):
    """Train PyTorch MLP model."""
    # ... 90 lines of PyTorch training code

def preprocess_tabular_data(X):
    """Preprocess tabular features."""
    # ... preprocessing logic
```

**Create:** `src/agents/model_trainer/nlp.py`

```python
"""
NLP Model Training
Transformer models with LoRA fine-tuning.
"""
from transformers import AutoModelForSequenceClassification, Trainer
from peft import LoraConfig, get_peft_model

async def train_transformer(data_path, target_column, context):
    """Train transformer model with optional LoRA."""
    # ... 180 lines of transformer training code
```

**Create:** `src/agents/model_trainer/vision.py`

```python
"""
Vision Model Training
CNN and other vision models.
"""

async def train_cnn(data_path, target_column, context):
    """Train CNN model."""
    raise NotImplementedError("Vision models coming soon")
```

**Create:** `src/agents/model_trainer/__init__.py`

```python
"""Model Trainer Module"""
from .trainer import ModelTrainerAgent

__all__ = ['ModelTrainerAgent']
```

---

### Step 4: Refactor SubmissionAgent

**Create:** `src/agents/submission/submitter.py`

```python
"""
Submission Agent
Main agent class for generating and submitting predictions.
"""
from agents.base import BaseAgent, AgentState
from .predictors import generate_predictions
from .formatters import auto_detect_format, format_submission

class SubmissionAgent(BaseAgent):
    """Agent responsible for submissions."""

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method."""

        # Generate predictions
        predictions = await generate_predictions(
            model_path, test_data_path, model_type, context
        )

        # Format submission
        submission_path = await format_submission(
            predictions, competition_name, context.get("format_spec")
        )

        # Submit to Kaggle
        if context.get("auto_submit"):
            result = await self._submit_to_kaggle(submission_path, ...)

        return self.results
```

**Create:** `src/agents/submission/predictors.py`

```python
"""
Prediction Generators
Generate predictions from trained models.
"""
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import torch

async def generate_predictions(model_path, test_data_path, model_type, context):
    """Generate predictions based on model type."""

    if model_type == "lightgbm":
        return await predict_lightgbm(model_path, test_data_path)
    elif model_type == "xgboost":
        return await predict_xgboost(model_path, test_data_path)
    # ...

async def predict_lightgbm(model_path, test_data_path):
    """Generate predictions with LightGBM."""
    # ... prediction logic

async def predict_xgboost(model_path, test_data_path):
    """Generate predictions with XGBoost."""
    # ... prediction logic

async def predict_pytorch_mlp(model_path, test_data_path, context):
    """Generate predictions with PyTorch MLP."""
    # ... prediction logic

async def predict_transformer(model_path, test_data_path, context):
    """Generate predictions with Transformers."""
    # ... prediction logic
```

**Create:** `src/agents/submission/formatters.py`

```python
"""
Submission Formatters
Auto-detect and format submission files.
"""
import pandas as pd
from pathlib import Path

async def auto_detect_format(competition_name, predictions):
    """Auto-detect submission format from sample file."""

    data_path = Path("data/raw") / competition_name
    sample_files = list(data_path.glob("*sample*.csv"))

    if sample_files:
        # Parse sample submission
        sample_df = pd.read_csv(sample_files[0], nrows=5)
        # ... detection logic

    return format_spec

async def format_submission(predictions, competition_name, format_spec):
    """Format predictions into submission file."""

    # Auto-detect if not specified
    if not format_spec:
        format_spec = await auto_detect_format(competition_name, predictions)

    # Apply formatting
    # ... formatting logic

    return submission_path
```

**Create:** `src/agents/submission/__init__.py`

```python
"""Submission Module"""
from .submitter import SubmissionAgent

__all__ = ['SubmissionAgent']
```

---

### Step 5: Refactor LeaderboardMonitorAgent

**Create:** `src/agents/leaderboard/monitor.py`

```python
"""
Leaderboard Monitor Agent
Main agent class for monitoring competition leaderboards.
"""
from agents.base import BaseAgent, AgentState
from .analyzer import analyze_performance, generate_recommendation

class LeaderboardMonitorAgent(BaseAgent):
    """Agent responsible for monitoring leaderboard."""

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method."""

        # Fetch leaderboard
        leaderboard_data = await self._fetch_leaderboard(competition_name)

        # Get current position
        current_rank, current_percentile = await self._find_position(
            leaderboard_data, username
        )

        # Analyze performance
        recommendation = generate_recommendation(
            current_percentile, self.target_percentile
        )

        return self.results
```

**Create:** `src/agents/leaderboard/analyzer.py`

```python
"""
Performance Analyzer
Analyze leaderboard performance and generate recommendations.
"""

def analyze_performance(current_percentile, target_percentile):
    """Analyze current performance vs target."""
    gap = current_percentile - target_percentile

    return {
        "gap": gap,
        "meets_target": current_percentile <= target_percentile,
        "gap_percentage": gap * 100
    }

def generate_recommendation(current_percentile, target_percentile):
    """Generate improvement recommendation."""

    gap = current_percentile - target_percentile

    if gap <= 0:
        return "excellent_performance"
    elif gap <= 0.05:
        return "maintain_performance"
    elif gap <= 0.10:
        return "minor_improvement_needed"
    elif gap <= 0.20:
        return "retrain_with_tuning"
    else:
        return "major_improvement_needed"
```

**Create:** `src/agents/leaderboard/__init__.py`

```python
"""Leaderboard Monitor Module"""
from .monitor import LeaderboardMonitorAgent

__all__ = ['LeaderboardMonitorAgent']
```

---

### Step 6: Refactor OrchestratorAgent

**Create:** `src/agents/orchestrator/orchestrator.py`

```python
"""
Orchestrator Agent
Main workflow coordination.
"""
from agents.base import BaseAgent, AgentState
from .strategies import try_different_model, improve_hyperparameters

class OrchestratorAgent(BaseAgent):
    """Central orchestrator for competition workflow."""

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main execution method."""

        # Phase 1: Data Collection
        data_results = await self._run_data_collection(context)

        # Phase 2: Model Training
        training_results = await self._run_initial_training(data_results, config)

        # Phase 3: Submission
        submission_results = await self._run_submission(training_results, data_results)

        # Phase 4: Optimization Loop
        final_results = await self._optimization_loop(data_results, training_results, context)

        return self.results
```

**Create:** `src/agents/orchestrator/strategies.py`

```python
"""
Optimization Strategies
Different strategies for improving performance.
"""

async def try_different_model(current_model, tried_models, data_results, context):
    """Try a different model type."""

    if current_model == "lightgbm" and "xgboost" not in tried_models:
        return await train_with_xgboost(data_results, context)
    elif current_model == "xgboost" and "lightgbm" not in tried_models:
        return await train_with_lightgbm(data_results, context)
    else:
        return None

async def improve_hyperparameters(config, current_percentile, target_percentile, aggressive=False):
    """Improve hyperparameters based on gap."""

    gap = current_percentile - target_percentile

    if aggressive or gap > 0.15:
        # Aggressive tuning
        improved_config = {
            "num_boost_round": config.get("num_boost_round", 1000) + 500,
            "learning_rate": config.get("learning_rate", 0.05) * 0.5,
            # ...
        }
    else:
        # Minor tuning
        improved_config = {
            "num_boost_round": config.get("num_boost_round", 1000) + 200,
            "learning_rate": config.get("learning_rate", 0.05) * 0.8,
            # ...
        }

    return improved_config
```

**Create:** `src/agents/orchestrator/__init__.py`

```python
"""Orchestrator Module"""
from .orchestrator import OrchestratorAgent

__all__ = ['OrchestratorAgent']
```

---

## Migration Script

Create a migration script to automate the refactoring:

**Create:** `scripts/refactor_agents.py`

```python
#!/usr/bin/env python3
"""
Agent Refactoring Script
Automatically migrates agents to modular structure.
"""
import shutil
from pathlib import Path

def refactor_agent(agent_name, splits):
    """Refactor a single agent into modules."""

    src_file = Path(f"src/agents/{agent_name}.py")
    dest_dir = Path(f"src/agents/{agent_name}")

    print(f"Refactoring {agent_name}...")

    # Create destination directory
    dest_dir.mkdir(exist_ok=True)

    # Split file according to plan
    for module_name, content_func in splits.items():
        module_file = dest_dir / f"{module_name}.py"
        content = content_func(src_file)
        module_file.write_text(content)

    # Create __init__.py
    init_file = dest_dir / "__init__.py"
    init_file.write_text(generate_init(agent_name))

    # Backup original file
    shutil.copy(src_file, src_file.with_suffix('.py.bak'))

    print(f"✓ {agent_name} refactored successfully!")

if __name__ == "__main__":
    # Run refactoring
    refactor_agent("data_collector", {...})
    refactor_agent("model_trainer", {...})
    # ...
```

---

## Benefits of Refactoring

### Before:
```python
# 600-line file - hard to navigate
# Hard to find specific functionality
# Difficult to test individual components
# Merge conflicts frequent
```

### After:
```python
# Small, focused files (50-200 lines each)
# Easy to find specific functionality
# Easy to test individual modules
# Fewer merge conflicts
# Better separation of concerns
```

---

## Testing After Refactoring

```bash
# 1. Verify imports work
python3 -c "from agents.base import BaseAgent; print('✓ Base')"
python3 -c "from agents.data_collector import DataCollectorAgent; print('✓ DataCollector')"
python3 -c "from agents.model_trainer import ModelTrainerAgent; print('✓ ModelTrainer')"

# 2. Run tests
pytest tests/test_agents/

# 3. Run full workflow
python3 kaggle_agent.py
```

---

## Rollback Plan

If anything breaks:

```bash
# Restore original files
cp src/agents/*.py.bak src/agents/*.py

# Remove new folders
rm -rf src/agents/*/

# System restored to working state!
```

---

## Summary

**Current:** 6 large files (2,301 lines)
**Target:** 20 focused files (same code, better organized)

**Benefits:**
- ✅ Easier to navigate
- ✅ Easier to test
- ✅ Easier to maintain
- ✅ Better separation of concerns
- ✅ Reduced file sizes

**Next Steps:**
1. Start with BaseAgent (simplest)
2. Then DataCollector
3. Then others one by one
4. Test after each migration

**Time Estimate:** 2-4 hours for complete refactoring

---

Ready to start? Begin with Step 1 (BaseAgent) as it's the smallest and easiest!