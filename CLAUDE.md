# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Project Vision

**Goal:** Build a universal multi-agent team that autonomously participates in ANY Kaggle competition and achieves top 20% ranking.

**Core Mission:**
1. **Collect Data** - Download competition data and external sources
2. **Train Models** - Select and train optimal models for any task type
3. **Submit Solutions** - Generate predictions and submit via Kaggle API
4. **Monitor Leaderboard** - Track performance and iterate until top 20% achieved

**Universal Capability:** The system must handle ANY Kaggle problem type:
- Tabular (regression, binary/multi-class classification, ranking)
- NLP (sentiment, QA, summarization, translation, generation)
- Computer Vision (classification, detection, segmentation)
- Time Series (forecasting, anomaly detection)
- Audio (speech recognition, classification)
- Multi-modal (image+text, video, etc.)
- Any competition format or evaluation metric

## 🏗️ Architecture Philosophy

### AI-First, Zero-Hardcoded Logic

**CRITICAL PRINCIPLE:** This system contains ZERO hardcoded assumptions about:
- Problem type or domain
- Data format or structure
- Target variable location or type
- Required preprocessing steps
- Model architecture selection
- Feature engineering strategies
- Hyperparameter values
- Competition-specific logic

**Everything is decided by AI agents** based on:
1. Reading competition problem statement
2. Understanding the goal and evaluation metric
3. Analyzing available data
4. Creating an execution plan
5. Adapting strategies based on leaderboard feedback

### Three-Tier Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Tier 1: AI AGENTS (Decision Makers)                    │
│  - Problem Understanding Agent                          │
│  - Planning Agent                                       │
│  - Data Analysis Agent                                  │
│  - Strategy Optimization Agent                          │
│  Uses: Google Gemini, Claude, GPT, etc.                │
└─────────────────────────────────────────────────────────┘
                          ↓ Decisions & Plans
┌─────────────────────────────────────────────────────────┐
│  Tier 2: ORCHESTRATOR (Workflow Manager)                │
│  - Executes AI-generated plans                          │
│  - Coordinates worker agents                            │
│  - Manages iteration loops                              │
│  - Handles error recovery                               │
└─────────────────────────────────────────────────────────┘
                          ↓ Tasks & Context
┌─────────────────────────────────────────────────────────┐
│  Tier 3: WORKERS (Task Executors)                       │
│  - Data Collector                                       │
│  - Model Trainer                                        │
│  - Submission Handler                                   │
│  - Leaderboard Monitor                                  │
└─────────────────────────────────────────────────────────┘
```

## 🔄 Competition Workflow

### Phase 1: Problem Understanding (AI-Driven)

**Objective:** Understand WHAT we're trying to solve before looking at data

**Steps:**
1. **Problem Understanding Agent** reads competition:
   - Competition description
   - Problem statement
   - Evaluation metric
   - Submission format requirements
   - Timeline and rules

2. **AI Analysis Output:**
   ```python
   {
       "competition_type": "tabular|nlp|vision|timeseries|audio|multimodal",
       "task_type": "regression|binary_classification|multiclass|detection|...",
       "evaluation_metric": "rmse|accuracy|f1|mAP|bleu|...",
       "success_criteria": "Description of what makes a good solution",
       "key_challenges": ["challenge1", "challenge2", ...],
       "recommended_approach": "High-level strategy"
   }
   ```

### Phase 2: Data Understanding (AI-Driven)

**Objective:** Analyze data in context of the problem

**Steps:**
1. **Data Collection Worker** downloads all competition files
2. **Data Analysis Agent** examines data with problem context:
   - File structure and formats
   - Data types and distributions
   - Target variable identification
   - Feature characteristics
   - Missing values and anomalies
   - External data opportunities

3. **AI Analysis Output:**
   ```python
   {
       "target_column": "column_name",
       "target_characteristics": {...},
       "data_quality": {...},
       "feature_types": {...},
       "external_data_needed": bool,
       "data_challenges": [...]
   }
   ```

### Phase 3: Planning (AI-Driven)

**Objective:** Create complete execution plan

**Steps:**
1. **Planning Agent** creates comprehensive plan:
   - Data preprocessing strategy
   - Feature engineering approaches
   - Model selection rationale
   - Training strategy
   - Validation approach
   - Ensemble methods

2. **AI Plan Output:**
   ```python
   {
       "preprocessing_steps": [
           {"step": "handle_missing", "strategy": "...", "reason": "..."},
           {"step": "feature_engineering", "features": [...], "reason": "..."},
           {"step": "encoding", "method": "...", "reason": "..."}
       ],
       "models_to_try": [
           {"model": "lightgbm", "priority": 1, "reason": "..."},
           {"model": "xgboost", "priority": 2, "reason": "..."}
       ],
       "validation_strategy": "5-fold CV with stratification",
       "ensemble_approach": "weighted average based on CV scores"
   }
   ```

### Phase 4: Execution (Worker-Driven)

**Objective:** Execute AI plan systematically

**Components:**

**4.1 Data Pipeline Executor:**
- Executes ALL preprocessing steps from AI plan
- Creates organized pipeline: `data/processed/{competition}/00_raw.csv` → `08_final.csv`
- Logs every transformation for reproducibility
- Adapts to any data modality (tabular/text/images/etc.)

**4.2 Model Trainer:**
- Trains models specified in AI plan
- Uses AI-recommended hyperparameters
- Implements AI-selected validation strategy
- Saves best checkpoints

**4.3 Submission Handler:**
- Generates predictions using trained model(s)
- Applies same preprocessing pipeline to test data
- Formats submission per competition requirements
- Submits via Kaggle API with tracking

### Phase 5: Monitoring & Iteration (AI-Driven)

**Objective:** Learn from results and improve

**Steps:**
1. **Leaderboard Monitor** fetches current ranking
2. **Strategy Optimization Agent** analyzes:
   - Current percentile vs target (20%)
   - Gap analysis
   - What's working / not working
   - Competitor analysis (if possible)

3. **AI Decision:**
   - If top 20%: Monitor and maintain
   - If not: Generate improved strategy and iterate

## 📁 Project Structure

```
src/
├── agents/
│   ├── base.py                          # BaseAgent for all workers
│   ├── llm_agents/                      # AI Decision Makers
│   │   ├── problem_understanding.py     # [PLANNED] Reads competition
│   │   ├── planning_agent.py            # [PLANNED] Creates execution plan
│   │   ├── data_analysis_agent.py       # [IMPLEMENTED] Analyzes data
│   │   └── strategy_optimizer.py        # [PLANNED] Improves based on feedback
│   ├── orchestrator/
│   │   ├── orchestrator.py              # [IMPLEMENTED] Workflow coordinator
│   │   └── phases.py                    # [IMPLEMENTED] Phase execution
│   ├── data_collector/
│   │   └── collector.py                 # [IMPLEMENTED] Downloads data
│   ├── model_trainer/
│   │   ├── trainer.py                   # [IMPLEMENTED] Trains models
│   │   └── data_pipeline.py             # [IMPLEMENTED] Universal preprocessing
│   ├── submission/
│   │   └── submitter.py                 # [IMPLEMENTED] Handles submissions
│   └── leaderboard/
│       └── monitor.py                   # [IMPLEMENTED] Tracks performance
├── models/                              # Model implementations
│   ├── tabular/                         # [IMPLEMENTED] LightGBM, XGBoost, MLP
│   ├── nlp/                             # [PARTIAL] Transformers with LoRA
│   └── vision/                          # [PLANNED] CNN, Vision Transformers
└── main.py                              # [IMPLEMENTED] Entry point
```

## 🚀 Current Implementation Status

### ✅ Fully Implemented
- BaseAgent architecture
- Orchestrator workflow coordination
- Data collection via Kaggle API
- AI-powered data analysis (Gemini)
- Universal data pipeline executor (8-step process)
- Tabular model training (LightGBM, XGBoost, PyTorch MLP)
- Submission handling with Kaggle API
- Leaderboard monitoring
- Iterative optimization loop

### 🟡 Partially Implemented
- NLP support (transformers with LoRA, needs more modalities)
- AI-driven feature engineering (executes AI recommendations)
- Problem understanding (currently only analyzes data, not problem statement)

### 🔴 Planned / To Be Implemented
- **Problem Understanding Agent** - Reads competition description first
- **Planning Agent** - Creates comprehensive execution plan
- **Strategy Optimization Agent** - Learns from leaderboard feedback
- **Computer Vision support** - Image classification, detection, segmentation
- **Time Series support** - Forecasting, anomaly detection
- **Audio support** - Speech, sound classification
- **Multi-modal support** - Image+text, video
- **Advanced ensembling** - Stacking, blending multiple models
- **AutoML integration** - H2O, AutoGluon for automated optimization
- **External data collection** - Automated search and integration

## 🔧 Key Implementation Principles

### 1. Universal Data Pipeline

The `DataPipelineExecutor` must handle ANY data type:

```python
async def execute(
    self,
    data_path: str,
    ai_plan: Dict[str, Any],  # Contains ALL preprocessing steps
    problem_context: Dict[str, Any]  # Competition type, metric, etc.
) -> ProcessedData:
    """
    Executes AI-generated preprocessing plan for ANY data modality.
    """
    # Detect data type from problem_context
    if problem_context["modality"] == "tabular":
        return await self._process_tabular(data_path, ai_plan)
    elif problem_context["modality"] == "nlp":
        return await self._process_text(data_path, ai_plan)
    elif problem_context["modality"] == "vision":
        return await self._process_images(data_path, ai_plan)
    # ... etc for all modalities
```

### 2. Universal Model Trainer

Model selection based ONLY on AI recommendations:

```python
async def train(self, context: Dict[str, Any]):
    """
    Trains model(s) specified in AI plan.
    No assumptions about task type.
    """
    ai_plan = context["ai_plan"]  # REQUIRED

    for model_spec in ai_plan["models_to_try"]:
        model = self._instantiate_model(model_spec)
        results = await self._train_with_ai_config(model, model_spec)

        if self._meets_criteria(results, ai_plan["success_criteria"]):
            return results
```

### 3. No Fallback Policy

**CRITICAL:** If AI fails, system fails. No hardcoded fallbacks.

```python
if not ai_plan:
    raise RuntimeError(
        "❌ No AI plan available. "
        "This is a pure agentic AI system - requires AI analysis. "
        "Check GEMINI_API_KEY or other LLM credentials."
    )
```

This ensures the system stays truly universal and doesn't degrade to hardcoded logic.

## 🌟 Development Guidelines

### When Adding New Features

**Always ask:**
1. ✅ Does this work for ANY Kaggle competition type?
2. ✅ Is the decision made by AI or hardcoded?
3. ✅ Can this adapt to unseen problem formats?
4. ✅ Does this require problem understanding first?

**Never:**
- ❌ Hardcode assumptions about data format
- ❌ Assume specific column names or types
- ❌ Use if/else chains for different competition types
- ❌ Create competition-specific code paths

**Instead:**
- ✅ Let AI analyze and decide
- ✅ Create generic executors that follow AI plans
- ✅ Design for unknown/future competition types
- ✅ Build flexible pipelines that adapt

### Code Organization

**AI Agents (Decision Makers):**
```python
# src/agents/llm_agents/
class ProblemUnderstandingAgent:
    """Reads and understands competition problem."""
    async def analyze_competition(self, competition_name: str) -> Dict:
        # Returns problem understanding, not data analysis
        pass

class PlanningAgent:
    """Creates execution plan based on problem + data."""
    async def create_plan(
        self,
        problem_context: Dict,
        data_context: Dict
    ) -> Dict:
        # Returns comprehensive execution plan
        pass
```

**Workers (Executors):**
```python
# src/agents/workers/
class UniversalPreprocessor:
    """Executes ANY preprocessing plan from AI."""
    async def execute_plan(self, plan: Dict) -> ProcessedData:
        # Adapts to any data type and plan
        pass
```

## 🔑 Environment Setup

### Required API Keys

```bash
# .env file
GEMINI_API_KEY=your-gemini-key          # For AI agents
KAGGLE_USERNAME=your-username            # For Kaggle API
KAGGLE_KEY=your-kaggle-key              # For Kaggle API

# Optional: Additional LLM providers
ANTHROPIC_API_KEY=your-claude-key       # For Claude agents
OPENAI_API_KEY=your-openai-key          # For GPT agents
```

### Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Run for any competition
python src/main.py

# The system will:
# 1. Read competition problem statement
# 2. Analyze data in context
# 3. Create execution plan
# 4. Train models
# 5. Submit and monitor
# 6. Iterate until top 20%
```

## 📊 Success Metrics

**Primary Goal:** Achieve top 20% on ANY Kaggle competition

**System Metrics:**
- Competition types successfully handled
- Average percentile ranking achieved
- Time to reach top 20%
- Automation level (% of decisions made by AI vs human)

**Quality Metrics:**
- Zero hardcoded competition-specific logic
- Successful handling of novel competition formats
- Adaptation to new data modalities
- Strategy improvement across iterations

## 🎯 Roadmap Priority

**Phase 1: Core Universal System** (Current)
- [x] Multi-agent architecture
- [x] AI-driven data analysis
- [x] Universal tabular pipeline
- [ ] Problem understanding first
- [ ] Comprehensive planning

**Phase 2: Multi-Modal Support**
- [ ] NLP competitions (full support)
- [ ] Computer vision competitions
- [ ] Time series competitions
- [ ] Audio competitions

**Phase 3: Advanced Optimization**
- [ ] AutoML integration
- [ ] Advanced ensembling
- [ ] Meta-learning from past competitions
- [ ] Automated external data discovery

**Phase 4: Full Autonomy**
- [ ] Zero human intervention required
- [ ] Automatic competition discovery
- [ ] Parallel competition participation
- [ ] Continuous learning system

## 💡 Remember

This documentation is the **north star** for all development. Every code change should move toward:
1. **More universal** - handles more competition types
2. **More intelligent** - AI makes more decisions
3. **Less hardcoded** - fewer assumptions in code
4. **More autonomous** - less human intervention needed

When in doubt, ask: "Will this work for a competition type we've never seen before?"