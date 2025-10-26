# Getting Started with Kaggle Multi-Agent System

## Quick Start

### 1. Set Up Environment

```bash
# Clone or navigate to the project
cd Finetuning_LLMs

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Kaggle API

```bash
# Create Kaggle directory
mkdir -p ~/.kaggle

# Download kaggle.json from https://www.kaggle.com/settings/account
# Move it to ~/.kaggle/

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

Alternatively, create a `.env` file:

```bash
# Copy the example
cp .env.example .env

# Edit .env and add your credentials
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
```

### 3. Run the System

#### Option 1: Interactive Menu (Recommended for beginners)

```bash
python src/main.py
```

This will show you a menu with options:
1. Run Full Competition Workflow (Automated)
2. Data Collection Only
3. Model Training Only
4. Check Leaderboard
5. Exit

#### Option 2: Programmatic Usage

```python
from agents import OrchestratorAgent
import asyncio

# Create orchestrator
orchestrator = OrchestratorAgent(
    competition_name="titanic",  # Replace with actual competition
    target_percentile=0.20,      # Top 20%
    max_iterations=5
)

# Run the full workflow
results = asyncio.run(orchestrator.run({}))

print(f"Final Rank: {results['final_rank']}")
print(f"Target Met: {results['target_met']}")
```

## Understanding the Workflow

The system runs in 4 main phases:

### Phase 1: Data Collection
- Downloads competition data from Kaggle
- Analyzes data structure, types, and statistics
- Identifies target column automatically
- Can fetch external data sources if provided

### Phase 2: Model Training
- Auto-detects task type (tabular/NLP/vision)
- Selects appropriate model (LightGBM, XGBoost, PyTorch, Transformers)
- Trains with validation split and early stopping
- Saves best model checkpoint

### Phase 3: Submission
- Generates predictions on test data
- Formats submission file per competition requirements
- Submits to Kaggle via API

### Phase 4: Optimization Loop
- Monitors leaderboard position
- Compares against target percentile
- Retrains with improved hyperparameters if needed
- Continues until target is met or max iterations reached

## Example: Running Your First Competition

```python
import asyncio
from agents import OrchestratorAgent

async def run_competition():
    # Initialize for Titanic competition
    orchestrator = OrchestratorAgent(
        competition_name="titanic",
        target_percentile=0.20,  # Top 20%
        max_iterations=3
    )

    # Configure training
    context = {
        "training_config": {
            "learning_rate": 0.05,
            "num_boost_round": 1000,
            "num_leaves": 31
        }
    }

    # Run it!
    results = await orchestrator.run(context)

    # Check results
    print(f"Final Rank: {results['final_rank']}")
    print(f"Final Percentile: {results['final_percentile']:.2%}")
    print(f"Target Met: {results['target_met']}")

    return results

# Run
asyncio.run(run_competition())
```

## Using Individual Agents

### Data Collection Only

```python
from agents import DataCollectorAgent
import asyncio

collector = DataCollectorAgent()
results = asyncio.run(collector.run({
    "competition_name": "titanic",
    "analyze": True
}))

print(results["analysis_report"])
```

### Training Only

```python
from agents import ModelTrainerAgent
import asyncio

trainer = ModelTrainerAgent()
results = asyncio.run(trainer.run({
    "data_path": "data/raw/titanic/train.csv",
    "target_column": "Survived",
    "config": {
        "learning_rate": 0.05,
        "num_boost_round": 1000
    }
}))

print(f"Model saved to: {results['model_path']}")
print(f"Best score: {results['best_score']}")
```

## Troubleshooting

### Kaggle API Issues

If you get authentication errors:
```bash
# Check if kaggle.json exists
ls -la ~/.kaggle/

# Verify permissions
chmod 600 ~/.kaggle/kaggle.json
```

### Module Not Found

If you get import errors:
```bash
# Make sure you're in the project root
cd Finetuning_LLMs

# Run with python -m
python -m src.main

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### GPU Not Available

The system works fine on CPU, but for transformer models:
```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

## Next Steps

1. **Try a simple competition** like Titanic or House Prices
2. **Experiment with configurations** in the training_config
3. **Monitor the logs** in `logs/kaggle_agent.log`
4. **Check saved models** in `models/final/`
5. **Review submissions** in `submissions/`

## Advanced Configuration

### Custom Training Config

```python
context = {
    "training_config": {
        # LightGBM/XGBoost
        "learning_rate": 0.05,
        "num_boost_round": 1000,
        "num_leaves": 31,
        "max_depth": -1,

        # PyTorch
        "batch_size": 64,
        "epochs": 100,
        "hidden_sizes": [128, 64, 32],

        # Transformers
        "model_name": "distilbert-base-uncased",
        "use_lora": True,
        "lora_r": 8,
        "max_length": 512
    }
}
```

### Adding External Data

```python
context = {
    "competition_name": "titanic",
    "external_sources": [
        "https://example.com/additional_data.csv",
        "https://example.com/more_features.json"
    ]
}
```

## Need Help?

- Check the [README.md](README.md) for architecture details
- Review [CLAUDE.md](CLAUDE.md) for development guidance
- Examine agent implementations in `src/agents/`
- Look at example usage in `src/main.py`