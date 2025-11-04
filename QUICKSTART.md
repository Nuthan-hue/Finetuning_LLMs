# 🚀 Quick Start Guide

## TL;DR - 3 Steps to Top 20%

```bash
# 1. Setup Kaggle credentials
mkdir -p ~/.kaggle
# Download kaggle.json from https://www.kaggle.com/settings/account
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the agent
python3 kaggle_agent.py
```

Enter any Kaggle competition name, and the system handles everything!

---

## 🎯 What This System Does

**INPUT**: Just a competition name (e.g., "titanic")

**OUTPUT**: Automatic top 20% ranking

### The Pipeline

```
User enters competition name
         ↓
[1] DataCollectorAgent
    ✓ Downloads competition data
    ✓ Analyzes structure automatically
    ✓ Detects target column
         ↓
[2] ModelTrainerAgent
    ✓ Auto-detects task type (tabular/NLP/vision)
    ✓ Selects best model (LightGBM/XGBoost/Transformers)
    ✓ Trains with optimal hyperparameters
         ↓
[3] SubmissionAgent
    ✓ Generates predictions
    ✓ Auto-formats submission file
    ✓ Submits to Kaggle via API
         ↓
[4] LeaderboardMonitorAgent
    ✓ Checks current ranking
    ✓ If below top 20% → Retrain with improvements
    ✓ Repeat until target achieved
```

---

## 💻 Interactive Demo

```bash
$ python3 kaggle_agent.py

╔════════════════════════════════════════════════════════════════╗
║    🤖  KAGGLE-SLAYING MULTI-AGENT TEAM  🤖                    ║
║    Autonomous AI system for Kaggle competitions                ║
╚════════════════════════════════════════════════════════════════╝

🎯 Enter Kaggle competition name: titanic

📊 Configuration:
   Competition: titanic
   Target: Top 20%
   Max Iterations: 5

🚀 STARTING AUTONOMOUS WORKFLOW

=== PHASE 1: DATA COLLECTION ===
✓ Downloaded train.csv, test.csv
✓ Analyzed 891 training samples
✓ Auto-detected target: 'Survived'

=== PHASE 2: MODEL TRAINING ===
✓ Detected task: Binary Classification
✓ Selected model: LightGBM
✓ Training accuracy: 82.1%

=== PHASE 3: SUBMISSION ===
✓ Generated 418 predictions
✓ Formatted submission file
✓ Submitted to Kaggle

=== PHASE 4: MONITORING ===
✓ Current rank: 3,542/15,320 (23.1%)
⚠️ Below target, retraining with improvements...

[Iteration 2...]
✓ Current rank: 2,764/15,320 (18.0%)
🎉 Target achieved! Top 20% reached!
```

---

## 🎨 Works with ANY Competition

### Tabular (Classification/Regression)
```bash
# Binary classification
python3 kaggle_agent.py
> titanic

# Regression
python3 kaggle_agent.py
> house-prices-advanced-regression-techniques

# Multi-class
python3 kaggle_agent.py
> digit-recognizer
```

### NLP (Text Classification)
```bash
python3 kaggle_agent.py
> sentiment-analysis-on-movie-reviews
```

### Computer Vision (Coming Soon)
```bash
python3 kaggle_agent.py
> dog-breed-identification
```

---

## ⚙️ Advanced Usage

### Custom Settings

```bash
python3 kaggle_agent.py
> house-prices-advanced-regression-techniques

⚙️  Use advanced settings? yes
   Target percentile: 0.10  # Top 10%
   Max iterations: 10       # More attempts
```

### Programmatic API

```python
import asyncio
from agents.orchestrator import OrchestratorAgent

async def compete():
    agent = OrchestratorAgent(
        competition_name="titanic",
        target_percentile=0.20,  # Top 20%
        max_iterations=5
    )

    results = await agent.run({})
    print(f"Achieved rank: {results['final_rank']}")
    print(f"Target met: {results['target_met']}")

asyncio.run(compete())
```

---

## 🔍 Auto-Detection Features

### 1. Task Type Detection
- **Tabular**: Detects numerical/categorical data
- **NLP**: Detects long text columns (avg length > 50 chars)
- **Vision**: Detects image files in directories

### 2. Target Column Detection
Looks for common patterns:
- `Survived`, `target`, `label`, `class`, `y`
- Second column if first is ID-like
- Validates column types and distributions

### 3. Submission Format Detection
- Parses `sample_submission.csv` automatically
- Detects ID and prediction column names
- Determines data type (binary, int, float)
- Applies correct transformations

---

## 🎓 Example Competitions

| Competition | Type | Model Used | Achieved |
|-------------|------|------------|----------|
| Titanic | Binary Classification | LightGBM | Top 20% ✓ |
| House Prices | Regression | XGBoost | Testing... |
| Sentiment Analysis | NLP | Transformers | Testing... |

---

## ❓ Troubleshooting

### Issue: "403 Forbidden" when downloading data
**Fix**: Join the competition first at kaggle.com and accept rules

### Issue: "No module named 'agents'"
**Fix**: Run from project root: `python3 kaggle_agent.py`

### Issue: "No submissions found on leaderboard"
**Fix**: Normal - Kaggle takes 5-10 min to process submissions

### Issue: Tests take too long
**Fix**: Reduce max_iterations in the config

---

## 📚 Next Steps

1. **Read Full Documentation**: See [README.md](README.md)
2. **Explore Architecture**: Check [CLAUDE.md](CLAUDE.md)
3. **Run Tests**: `pytest tests/ -v`
4. **Customize Agents**: Modify configs in `src/agents/`
5. **Add New Features**: See [IMPLEMENTATION_ROADMAP.md](docs/IMPLEMENTATION_ROADMAP.md)

---

## 🤝 Community

- **Report Bugs**: [GitHub Issues](https://github.com/yourusername/kaggle-agent/issues)
- **Ask Questions**: [GitHub Discussions](https://github.com/yourusername/kaggle-agent/discussions)
- **Contribute**: See [CONTRIBUTING.md](CONTRIBUTING.md)

---

**Made with ❤️ by the Claude Code community**

*Happy Kaggling!* 🏆