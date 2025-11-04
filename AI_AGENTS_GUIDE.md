# 🤖 AI Agents Guide

## Overview

This system now supports **TRUE AI AGENTS** powered by Google Gemini, replacing hardcoded conditional logic with intelligent reasoning.

## 🔄 Before vs After

### ❌ Before: Hardcoded Rules (Old Approach)

```python
# orchestrator/strategies.py - HARDCODED CONDITIONALS
def select_optimization_strategy(recommendation, current_model, ...):
    if recommendation == "minor_improvement_needed":
        return {"action": "retrain", "aggressive": False}
    elif recommendation == "major_improvement_needed":
        return {"action": "retrain", "aggressive": True}
    else:
        if current_model == "lightgbm":
            return {"action": "switch_model", "new_model": "xgboost"}
        elif current_model == "xgboost":
            return {"action": "switch_model", "new_model": "lightgbm"}
```

**Problems:**
- Fixed rules can't adapt to unique situations
- No reasoning about competition-specific strategies
- Can't learn from performance trends
- Limited to pre-programmed logic

### ✅ After: AI-Powered Agents (New Approach)

```python
# llm_agents/strategy_agent.py - INTELLIGENT REASONING
strategy_agent = StrategyAgent()  # Uses Gemini AI

strategy = await strategy_agent.select_optimization_strategy(
    recommendation=recommendation,
    current_model="lightgbm",
    tried_models=["lightgbm"],
    current_percentile=0.45,
    target_percentile=0.20,
    iteration=2,
    competition_type="tabular",
    performance_history=[...]
)

# AI returns:
{
    "action": "tune_aggressive",
    "reasoning": "Current model shows promise but needs stronger regularization.
                  Gap of 25% requires significant improvement. Recommend increasing
                  num_boost_round to 2000 and reducing learning_rate to 0.01 for
                  better generalization...",
    "config_updates": {
        "num_boost_round": 2000,
        "learning_rate": 0.01,
        "max_depth": 8,
        "min_child_weight": 3
    },
    "expected_improvement": "10-15% percentile improvement",
    "confidence": "high"
}
```

**Benefits:**
- ✅ Adapts to each unique situation
- ✅ Provides detailed reasoning
- ✅ Learns from performance history
- ✅ Considers competition-specific factors
- ✅ Suggests specific hyperparameter values

---

## 🚀 Setup

### 1. Install Google AI SDK

```bash
pip install google-generativeai python-dotenv
```

### 2. Get Gemini API Key

1. Go to https://makersuite.google.com/app/apikey
2. Create a new API key
3. Add to `.env` file:

```bash
# .env
GEMINI_API_KEY=your_api_key_here
```

### 3. Verify Installation

```python
from src.agents.llm_agents import StrategyAgent

agent = StrategyAgent()
print("✓ AI Agent ready!")
```

---

## 🤖 Available AI Agents

### 1. **StrategyAgent** - Intelligent Optimization Decisions

Replaces hardcoded `select_optimization_strategy()` with AI reasoning.

**What it does:**
- Analyzes current performance vs target
- Reviews what models have been tried
- Examines performance trends
- Recommends next optimization strategy
- Provides detailed reasoning

**Usage:**

```python
from src.agents.llm_agents import StrategyAgent

strategy_agent = StrategyAgent()

strategy = await strategy_agent.select_optimization_strategy(
    recommendation="major_improvement_needed",
    current_model="lightgbm",
    tried_models=["lightgbm"],
    current_percentile=0.45,
    target_percentile=0.20,
    iteration=2,
    competition_type="tabular",
    performance_history=workflow_history
)

print(f"AI Decision: {strategy['action']}")
print(f"Reasoning: {strategy['reasoning']}")
print(f"Expected Improvement: {strategy['expected_improvement']}")
```

### 2. **DataAnalysisAgent** - Intelligent Data Understanding

Analyzes datasets and suggests preprocessing strategies.

**What it does:**
- Identifies target variable
- Detects data quality issues
- Suggests preprocessing strategies
- Recommends feature engineering
- Warns about potential problems

**Usage:**

```python
from src.agents.llm_agents import DataAnalysisAgent

data_agent = DataAnalysisAgent()

analysis = await data_agent.analyze_and_suggest(
    dataset_info={
        "datasets": {
            "train.csv": {
                "rows": 891,
                "columns": ["PassengerId", "Survived", "Pclass", ...],
                "dtypes": {...},
                "missing_values": {...}
            }
        }
    },
    competition_name="titanic"
)

print(f"Target: {analysis['target_column']}")
print(f"Recommended Models: {analysis['recommended_models']}")
print(f"Feature Engineering: {analysis['feature_engineering']}")
```

---

## 🎯 Using AI Agents in Orchestrator

### Option 1: AI-Powered Mode (Recommended)

```python
from agents.orchestrator import OrchestratorAgent

# Enable AI agents by importing AI optimization
from agents.orchestrator.optimization_ai import run_optimization_loop_ai

orchestrator = OrchestratorAgent(
    competition_name="titanic",
    target_percentile=0.20,
    max_iterations=5
)

# Override optimization method to use AI
orchestrator._optimization_loop = lambda *args, **kwargs: run_optimization_loop_ai(
    orchestrator, *args, **kwargs
)

# Run with AI decision-making
results = await orchestrator.run({})
```

### Option 2: Hybrid Mode

Use AI for some decisions, hardcoded for others:

```python
# In orchestrator/optimization.py

# Check if AI is available
try:
    from ..llm_agents import StrategyAgent
    strategy_agent = StrategyAgent()
    USE_AI = True
except Exception:
    USE_AI = False
    logger.warning("AI agents not available, using fallback logic")

# Use AI if available
if USE_AI:
    strategy = await strategy_agent.select_optimization_strategy(...)
else:
    strategy = select_optimization_strategy(...)  # Fallback to hardcoded
```

---

## 📊 AI Agent Outputs

### Strategy Agent Output

```json
{
    "action": "switch_model",
    "reasoning": "LightGBM has plateaued after 2 iterations. XGBoost typically performs better on this type of categorical-heavy dataset. The 25% gap requires trying alternative approaches.",
    "new_model": "xgboost",
    "aggressive": false,
    "config_updates": {
        "model_type": "xgboost",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 1500,
        "subsample": 0.8
    },
    "expected_improvement": "5-10% percentile improvement",
    "confidence": "high"
}
```

### Data Analysis Agent Output

```json
{
    "target_column": "Survived",
    "target_confidence": "high",
    "task_type": "binary_classification",
    "data_quality": {
        "missing_values_strategy": "Age: use median, Cabin: drop (70% missing)",
        "outliers": "Fare has outliers, consider log transformation",
        "class_balance": "slightly imbalanced (62% vs 38%)"
    },
    "preprocessing": {
        "categorical_encoding": "onehot for Sex/Embarked, ordinal for Pclass",
        "numerical_scaling": "standard",
        "handle_missing": "median for Age, mode for Embarked",
        "feature_transformations": ["log(Fare)", "binning(Age into age_groups)"]
    },
    "feature_engineering": [
        "family_size = SibSp + Parch + 1",
        "is_alone = 1 if family_size == 1 else 0",
        "title extraction from Name (Mr, Mrs, Miss, etc.)",
        "fare_per_person = Fare / family_size",
        "age_class interaction = Age * Pclass"
    ],
    "warnings": [
        "PassengerId is just an ID, should be dropped",
        "Cabin has 77% missing values, consider dropping",
        "Name contains useful info (titles), extract before dropping"
    ],
    "recommended_models": ["lightgbm", "xgboost", "random_forest"],
    "confidence": "high"
}
```

---

## 🔧 Configuration

### Adjust AI Behavior

```python
# Change model (faster vs more capable)
strategy_agent = StrategyAgent()
strategy_agent.model_name = "gemini-1.5-pro"  # More capable
strategy_agent.model_name = "gemini-1.5-flash"  # Faster (default)

# Adjust creativity
strategy_agent.temperature = 0.9  # More creative
strategy_agent.temperature = 0.3  # More focused (default for data analysis)
```

### Custom System Prompts

```python
from src.agents.llm_agents import BaseLLMAgent

custom_agent = BaseLLMAgent(
    name="CustomAgent",
    model_name="gemini-1.5-flash",
    temperature=0.7,
    system_prompt="""You are an expert in XGBoost optimization.
    Always recommend XGBoost-specific hyperparameters."""
)

response = await custom_agent.reason(
    "What hyperparameters should I tune for XGBoost?"
)
```

---

## 💡 Best Practices

1. **Start with AI, fallback to hardcoded**: Use try/except to fallback if AI fails
2. **Log AI reasoning**: Always log the AI's reasoning for debugging
3. **Validate AI outputs**: Check that returned JSON has required fields
4. **Set appropriate temperatures**:
   - 0.1-0.3: Factual tasks (data analysis)
   - 0.5-0.7: Balanced reasoning (strategy selection)
   - 0.8-1.0: Creative tasks (feature ideas)
5. **Monitor API costs**: Gemini Flash is cheap but still costs money

---

## 🎓 Example: Full AI-Powered Workflow

```python
import asyncio
from agents.orchestrator import OrchestratorAgent
from agents.llm_agents import StrategyAgent, DataAnalysisAgent

async def main():
    # Initialize agents
    orchestrator = OrchestratorAgent(
        competition_name="titanic",
        target_percentile=0.20
    )

    # Run with AI-powered optimization
    results = await orchestrator.run({})

    print(f"Final rank: {results['final_rank']}")
    print(f"Total iterations: {results['total_iterations']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🆚 Comparison Summary

| Feature | Hardcoded Rules | AI Agents |
|---------|----------------|-----------|
| Decision Quality | Fixed logic | Contextual reasoning |
| Adaptability | None | High |
| Explanations | None | Detailed reasoning |
| Learning | No | From history |
| Competition-specific | Generic | Tailored |
| Setup Complexity | None | API key needed |
| Cost | Free | ~$0.01 per competition |
| Reliability | 100% | 95% (with fallback) |

---

## 🐛 Troubleshooting

### "GEMINI_API_KEY not found"

Add to `.env`:
```bash
GEMINI_API_KEY=your_key_here
```

### "google-generativeai not installed"

```bash
pip install google-generativeai
```

### AI returns invalid JSON

The system has fallback logic. Check logs for reasoning:
```
WARNING: Using fallback strategy due to AI error
```

### API rate limits

Gemini Flash: 15 requests/minute (plenty for competitions)

---

## 🚀 Next Steps

1. ✅ **Integrated**: Strategy Agent, Data Analysis Agent
2. 🚧 **In Progress**: Model Selection Agent, Hyperparameter Tuning Agent
3. 📋 **Planned**: Feature Engineering Agent, Ensemble Agent

The system now uses TRUE AI agents for intelligent decision-making! 🎉