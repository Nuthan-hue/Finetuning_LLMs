# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Project Vision

**Goal:** Build a universal multi-agent system that autonomously participates in ANY Kaggle competition and achieves top 20% ranking.

**Core Mission:**
1. **Understand Problem** - Read competition description and verify against data
2. **Analyze Data** - Deep analysis with preprocessing recommendations
3. **Clean & Engineer** - Prepare data with AI-generated code
4. **Plan Strategy** - Select models and approaches
5. **Train Models** - Execute training with optimal configurations
6. **Submit & Iterate** - Submit predictions and improve until target achieved

**Universal Capability:** The system architecture handles ANY Kaggle problem type:
- ✅ **Tabular** (regression, binary/multi-class classification, ranking) - FULLY IMPLEMENTED
- ✅ **NLP** (sentiment, classification, QA, generation) - FULLY IMPLEMENTED
- 🏗️ **Computer Vision** (classification, detection, segmentation) - ARCHITECTURE READY
- 🏗️ **Time Series** (forecasting, anomaly detection) - ARCHITECTURE READY
- 🏗️ **Audio** (speech recognition, classification) - ARCHITECTURE READY
- 🏗️ **Multi-modal** (image+text, video, etc.) - ARCHITECTURE READY

---

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

### Sequential Pipeline with Conditional Agents

**Pattern:** Sequential flow with conditional agent invocation
- Easy to understand and debug
- Cost-efficient (skip unnecessary agents)
- Predictable execution
- Optimized for learning and development

**Flow:**
```
Always Called → DataCollector
Always Called → ProblemUnderstandingAgent
Always Called → DataAnalysisAgent
Conditional  → PreprocessingAgent (only if needs_preprocessing)
Always Called → PlanningAgent
Conditional  → FeatureEngineeringAgent (only if needs_feature_engineering)
Always Called → ModelTrainer
Always Called → EvaluationAgent
Conditional  → StrategyOptimizer (only if not at target, loops back)
```

---

## 📊 Multi-Agent Architecture (Option B: Core Modalities)

### Implementation Strategy

**By Nov 27, 2024:**
- ✅ **Tabular Competitions:** Fully implemented (LightGBM, XGBoost, PyTorch MLP)
- ✅ **NLP Competitions:** Fully implemented (BERT, Transformers)
- 📋 **Vision/Time Series:** Architecture ready, implementations pending

**Why Option B?**
- Covers 70% of Kaggle competitions (tabular + NLP)
- Realistic for 22-day timeline
- Demonstrates universal architecture
- Clear extension path for other modalities

---

## 🔄 Agent Flow (10 Phases)

```
┌─────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│  - Coordinates workflow (no AI, just management)        │
│  - Passes context between agents                        │
│  - Conditionally invokes agents based on flags          │
└─────────────────────────────────────────────────────────┘

Phase 1: DATA COLLECTION (Worker - No LLM Cost)
─────────────────────────────────────────────────────────
│ Downloads: train.csv, test.csv, problem description
│ Basic analysis: file sizes, row/column counts
└─→ Output: Raw files + basic statistics

Phase 2: PROBLEM UNDERSTANDING (LLM Agent 🤖)
─────────────────────────────────────────────────────────
│ Input: Competition text + Raw data files
│ Reads: Problem description AND verifies against data
│ Output: Problem context (task type, metric, target)
└─→ 💰 Cost: 1 LLM call

Phase 3: DATA ANALYSIS (LLM Agent 🤖)
─────────────────────────────────────────────────────────
│ Input: Problem context + Raw data
│ Analyzes: Missing values, types, distributions, correlations
│ Detects: Data modality (tabular/nlp/vision/timeseries)
│ Decides: needs_preprocessing? (true/false)
│ Output: Analysis report + preprocessing recommendations
└─→ 💰 Cost: 1 LLM call

Phase 4: PREPROCESSING (Conditional - LLM + Worker)
─────────────────────────────────────────────────────────
IF needs_preprocessing == True:
  │ PreprocessingAgent (LLM 🤖):
  │   - Reads DataAnalysis recommendations
  │   - Writes Python preprocessing code (modality-aware)
  │   - Returns executable code
  │
  │ Executor (Worker):
  │   - Executes generated code
  │   - Saves clean_train.csv, clean_test.csv
  └─→ Output: Clean data (0% missing, encoded, normalized)
      💰 Cost: 1 LLM call (if preprocessing needed)
ELSE:
  └─→ Skip (0 LLM calls, use raw data)

Phase 5: PLANNING (LLM Agent 🤖)
─────────────────────────────────────────────────────────
│ Input: Problem + Analysis + Clean data
│ Creates: Model strategy, hyperparameters, validation plan
│ Decides: needs_feature_engineering? (true/false)
│ Output: Execution plan with model configs
└─→ 💰 Cost: 1 LLM call

Phase 6: FEATURE ENGINEERING (Conditional - LLM + Worker)
─────────────────────────────────────────────────────────
IF needs_feature_engineering == True:
  │ FeatureEngineeringAgent (LLM 🤖):
  │   - Reads PlanningAgent recommendations
  │   - Writes Python feature engineering code
  │   - Returns executable code
  │
  │ Executor (Worker):
  │   - Executes generated code on clean data
  │   - Saves featured_train.csv
  └─→ Output: Featured data
      💰 Cost: 1 LLM call (if features needed)
ELSE:
  └─→ Skip (0 LLM calls, use clean data)

Phase 7: MODEL TRAINING (Worker - No LLM Cost)
─────────────────────────────────────────────────────────
│ Input: Featured/clean data + Execution plan
│ Trains: Models specified in plan (LightGBM, XGBoost, BERT, etc.)
│ Uses: Hyperparameters from plan
│ Validation: Strategy from plan (stratified k-fold, etc.)
│ Output: Trained models + CV scores
└─→ 💰 Cost: 0 (pure execution)

Phase 8: SUBMISSION (Worker - No LLM Cost)
─────────────────────────────────────────────────────────
│ Generates: Predictions on test data
│ Formats: Per competition requirements
│ Submits: To Kaggle via API
│ Output: Submission file + Leaderboard score
└─→ 💰 Cost: 0

Phase 9: EVALUATION (LLM Agent 🤖)
─────────────────────────────────────────────────────────
│ Input: CV scores, LB score, training metrics
│ Analyzes: CV vs LB gap, overfitting, underfitting
│ Diagnoses: Issues and hypotheses
│ Decides: needs_improvement? (true/false)
│ Output: Diagnosis report
└─→ 💰 Cost: 1 LLM call

Phase 10: OPTIMIZATION (Conditional - LLM Agent 🤖)
─────────────────────────────────────────────────────────
IF needs_improvement == True AND iteration < max_iterations:
  │ StrategyOptimizer (LLM 🤖):
  │   - Reads evaluation diagnosis
  │   - Suggests specific changes
  │   - Decides where to loop back (Phase 4, 5, or 6)
  │   - Returns optimization strategy
  └─→ Loop back to appropriate phase
      💰 Cost: 1 LLM call per iteration
ELSE:
  └─→ Done! Target achieved or max iterations reached
```

---

## 📋 Agent Communication Table

| Phase | Agent | Type | Says What | To Whom | LLM Cost |
|-------|-------|------|-----------|---------|----------|
| **1** | DataCollector | ⚙️ Worker | "Downloaded train.csv (891×12), test.csv (418×11), problem.txt" | → ProblemUnderstanding | **Free** |
| **2** | ProblemUnderstandingAgent | 🤖 LLM | "Binary classification. Target: Survived. Metric: Accuracy. Problem-data aligned ✓" | → DataAnalysis | **1 call** |
| **3** | DataAnalysisAgent | 🤖 LLM | "Modality: tabular. Age: 20% missing. Sex needs encoding. **needs_preprocessing: true**" | → PreprocessingAgent | **1 call** |
| **4a** | PreprocessingAgent | 🤖 LLM | "Generated preprocessing code: [impute Age median, encode Sex/Pclass, drop Cabin]" | → Executor | **1 call** (conditional) |
| **4b** | Executor | ⚙️ Worker | "Preprocessing complete ✓. Output: clean_train.csv (891×9, 0% missing)" | → PlanningAgent | **Free** |
| **5** | PlanningAgent | 🤖 LLM | "Strategy: LightGBM (priority 1), XGBoost (priority 2). **needs_feature_engineering: true**" | → FeatureEngineeringAgent | **1 call** |
| **6a** | FeatureEngineeringAgent | 🤖 LLM | "Generated feature code: [family_size, is_alone, age_bins, title]" | → Executor | **1 call** (conditional) |
| **6b** | Executor | ⚙️ Worker | "Features created ✓. Output: featured_train.csv (891×13)" | → ModelTrainer | **Free** |
| **7** | ModelTrainer | ⚙️ Worker | "LightGBM CV: 0.815. XGBoost CV: 0.808. Ensemble CV: 0.823" | → Submitter | **Free** |
| **8** | Submitter | ⚙️ Worker | "Submitted to Kaggle. Leaderboard score: 0.79. Rank: 30th percentile" | → EvaluationAgent | **Free** |
| **9** | EvaluationAgent | 🤖 LLM | "CV: 0.823, LB: 0.79. Gap: 3.3% (overfitting). **needs_improvement: true**" | → StrategyOptimizer | **1 call** |
| **10** | StrategyOptimizer | 🤖 LLM | "Add L1/L2 regularization. Drop title feature. Loop back to Phase 6" | → FeatureEngineeringAgent (iter 2) | **1 call** (conditional) |

**Iteration 1 Total: ~7 LLM calls** (if both preprocessing and features needed)

---

## 💰 Cost Analysis (Gemini Free Tier Friendly)

### Scenario 1: Titanic (Tabular with preprocessing + features)
```
✅ ProblemUnderstanding     → 1 call
✅ DataAnalysis             → 1 call
✅ PreprocessingAgent       → 1 call (needed)
✅ PlanningAgent            → 1 call
✅ FeatureEngineeringAgent  → 1 call (needed)
✅ EvaluationAgent          → 1 call
✅ StrategyOptimizer        → 1 call (if not at target)
─────────────────────────────
Total: 7 LLM calls per iteration
```

### Scenario 2: Clean UCI Dataset (No preprocessing)
```
✅ ProblemUnderstanding     → 1 call
✅ DataAnalysis             → 1 call (says needs_preprocessing: false)
⏭️  PreprocessingAgent       → 0 calls (SKIPPED)
✅ PlanningAgent            → 1 call
✅ FeatureEngineeringAgent  → 1 call
✅ EvaluationAgent          → 1 call
✅ StrategyOptimizer        → 1 call
─────────────────────────────
Total: 6 LLM calls
```

### Scenario 3: Image Classification (No preprocessing/features)
```
✅ ProblemUnderstanding     → 1 call
✅ DataAnalysis             → 1 call (says needs_preprocessing: false)
⏭️  PreprocessingAgent       → 0 calls (SKIPPED)
✅ PlanningAgent            → 1 call (says needs_feature_engineering: false)
⏭️  FeatureEngineeringAgent  → 0 calls (SKIPPED)
✅ EvaluationAgent          → 1 call
✅ StrategyOptimizer        → 1 call
─────────────────────────────
Total: 5 LLM calls
```

**Gemini Free Tier:** 60 requests/minute
**Safe for:** Multiple iterations, experimentation, learning ✅

---

## 🎯 Agent Specifications

### 1. DataCollector (Worker)
**Role:** Downloads competition files
**Input:** `competition_name: str`
**Output:**
```python
{
  "data_path": "/data/titanic/",
  "files": ["train.csv", "test.csv", "sample_submission.csv"],
  "problem_description": "text from Kaggle page",
  "basic_stats": {"train.csv": {"rows": 891, "columns": 12}}
}
```
**Cost:** Free (no LLM)

---

### 2. ProblemUnderstandingAgent (LLM)
**Role:** Understands competition by reading problem text AND verifying against data
**Input:**
- `problem_description: str`
- `data_files: List[str]`
- `basic_stats: Dict`

**Output:**
```python
{
  "competition_type": "binary_classification",
  "task_description": "Predict passenger survival on Titanic",
  "evaluation_metric": "accuracy",
  "submission_format": {
    "id_column": "PassengerId",
    "prediction_column": "Survived",
    "output_type": "binary"
  },
  "data_alignment": {
    "problem_claims": "Predict survival",
    "data_confirms": "Survived column exists (0/1)",
    "matches": true
  },
  "timeline": "30 days",
  "key_challenges": [
    "Small dataset (891 rows)",
    "Missing values visible",
    "Imbalanced target possible"
  ]
}
```
**Cost:** 1 LLM call

---

### 3. DataAnalysisAgent (LLM)
**Role:** Deep data analysis with modality detection and preprocessing recommendations
**Input:**
- `problem_understanding: Dict`
- `data_path: str`
- `files: List[str]`

**Output:**
```python
{
  "data_modality": "tabular",  # ← CRITICAL for routing
  "target_column": "Survived",
  "target_type": "binary",
  "target_distribution": {"0": 549, "1": 342},
  "is_imbalanced": true,

  "feature_types": {
    "id_columns": ["PassengerId"],
    "numerical": ["Age", "Fare", "SibSp", "Parch"],
    "categorical": ["Sex", "Pclass", "Embarked"],
    "text": ["Name"],
    "drop_candidates": ["PassengerId", "Ticket", "Cabin"]
  },

  "data_quality": {
    "missing_values": {
      "Age": {"count": 177, "percentage": 0.20},
      "Cabin": {"count": 687, "percentage": 0.77}
    },
    "outliers": ["Fare"],
    "class_balance": "imbalanced"
  },

  "preprocessing_required": true,  # ← DECISION FLAG
  "preprocessing_recommendations": {
    "modality": "tabular",
    "drop_columns": ["PassengerId", "Ticket", "Cabin"],
    "impute_missing": {
      "Age": {"method": "median", "reason": "normally distributed"},
      "Embarked": {"method": "mode", "reason": "only 2 missing"}
    },
    "encode_categorical": {
      "Sex": "label",
      "Pclass": "label",
      "Embarked": "label"
    },
    "handle_outliers": {
      "Fare": {"method": "cap", "percentile": 99}
    }
  }
}
```
**Cost:** 1 LLM call

---

### 4. PreprocessingAgent (LLM) - Conditional
**Role:** Generates executable Python code for data preprocessing
**Input:**
- `data_analysis: Dict` (with preprocessing_recommendations)
- `data_modality: str` (tabular/nlp/vision/timeseries)
- `raw_data_path: str`

**Output:**
```python
{
  "preprocessing_code": """
import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    # Load raw data
    df = pd.read_csv(input_path)

    # Drop useless columns
    df = df.drop(columns=['PassengerId', 'Ticket', 'Cabin'])

    # Impute Age with median
    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)

    # Encode Sex
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    # Cap Fare outliers
    fare_cap = df['Fare'].quantile(0.99)
    df['Fare'] = df['Fare'].clip(upper=fare_cap)

    # Save clean data
    df.to_csv(output_path, index=False)
    return len(df), len(df.columns)
""",
  "explanation": "Drops IDs, imputes Age median, encodes Sex, caps Fare",
  "expected_output": "clean_train.csv (891 rows, 9 columns, 0% missing)"
}
```
**Cost:** 1 LLM call (only if preprocessing_required == true)

**Modality-Aware Prompting:**
- **Tabular:** Impute, encode, scale, outliers
- **NLP:** Lowercase, remove URLs, tokenize, remove stopwords
- **Vision:** Normalize, resize, augmentation
- **Time Series:** Parse dates, create time index, handle gaps

---

### 5. PlanningAgent (LLM)
**Role:** Creates comprehensive strategy with model selection
**Input:**
- `problem_understanding: Dict`
- `data_analysis: Dict`
- `clean_data_path: str` (or raw if no preprocessing)
- `clean_data_stats: Dict`

**Output:**
```python
{
  "strategy_summary": "Tree-based models with engineered features",
  "data_modality": "tabular",  # Pass through for downstream

  "models_to_try": [
    {
      "model": "lightgbm",
      "priority": 1,
      "reason": "Fast, handles missing values, great for tabular",
      "hyperparameters": {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "max_depth": 7
      },
      "expected_performance": "0.78-0.82 accuracy"
    },
    {
      "model": "xgboost",
      "priority": 2,
      "reason": "Often beats LightGBM, ensemble candidate",
      "hyperparameters": {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 100
      }
    }
  ],

  "feature_engineering_required": true,  # ← DECISION FLAG
  "feature_engineering_plan": [
    {
      "feature_name": "family_size",
      "formula": "SibSp + Parch + 1",
      "reason": "Capture family unit effect",
      "priority": 1
    },
    {
      "feature_name": "age_bins",
      "formula": "pd.cut(Age, bins=[0,12,18,35,60,100])",
      "reason": "Age is clean now, can bin safely",
      "priority": 1
    }
  ],

  "validation_strategy": {
    "method": "stratified_kfold",
    "n_splits": 5,
    "stratify_column": "Survived",
    "shuffle": true,
    "random_state": 42,
    "reason": "Preserve class ratio, reduce variance"
  },

  "success_criteria": {
    "target_metric_value": 0.80,
    "target_percentile": 0.20,
    "max_training_time_hours": 1
  }
}
```
**Cost:** 1 LLM call

---

### 6. FeatureEngineeringAgent (LLM) - Conditional
**Role:** Generates executable Python code for feature engineering
**Input:**
- `feature_engineering_plan: List[Dict]` (from PlanningAgent)
- `clean_data_path: str`
- `data_modality: str`

**Output:**
```python
{
  "feature_engineering_code": """
import pandas as pd

def engineer_features(input_path, output_path):
    df = pd.read_csv(input_path)

    # Feature 1: family_size
    df['family_size'] = df['SibSp'] + df['Parch'] + 1

    # Feature 2: is_alone
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    # Feature 3: age_bins
    df['age_bins'] = pd.cut(
        df['Age'],
        bins=[0, 12, 18, 35, 60, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    df.to_csv(output_path, index=False)
    return len(df), len(df.columns)
""",
  "explanation": "Creates 3 features: family_size, is_alone, age_bins",
  "expected_output": "featured_train.csv (891 rows, 12 columns)"
}
```
**Cost:** 1 LLM call (only if feature_engineering_required == true)

---

### 7. ModelTrainer (Worker)
**Role:** Executes training based on execution plan
**Input:**
- `execution_plan: Dict` (from PlanningAgent)
- `data_path: str` (featured or clean data)
- `target_column: str`

**Actions:**
1. Loads data from data_path
2. For each model in `models_to_try`:
   - Instantiates model with hyperparameters from plan
   - Sets up validation per `validation_strategy`
   - Trains model
   - Tracks CV scores
3. Saves trained models
4. Returns results

**Output:**
```python
{
  "models_trained": [
    {
      "model_type": "lightgbm",
      "model_path": "/models/titanic/lightgbm_fold_avg.pkl",
      "cv_score": 0.815,
      "cv_std": 0.023,
      "fold_scores": [0.82, 0.81, 0.83, 0.80, 0.81],
      "training_time": 12.3
    },
    {
      "model_type": "xgboost",
      "cv_score": 0.808,
      "cv_std": 0.028
    }
  ],
  "best_model": "lightgbm",
  "ensemble_score": 0.823
}
```
**Cost:** Free (no LLM)

**Modality Routing:**
```python
if modality == "tabular":
    if model_name == "lightgbm": return train_lightgbm(...)
    elif model_name == "xgboost": return train_xgboost(...)
elif modality == "nlp":
    if "bert" in model_name: return train_bert(...)
elif modality == "vision":
    raise NotImplementedError("Vision models coming soon")
```

---

### 8. Submitter (Worker)
**Role:** Generates predictions and submits to Kaggle
**Input:**
- `model_path: str`
- `test_data_path: str`
- `submission_format: Dict` (from ProblemUnderstanding)

**Actions:**
1. Applies same preprocessing/features to test data
2. Loads trained model
3. Generates predictions
4. Formats per submission requirements
5. Submits via Kaggle API

**Output:**
```python
{
  "submission_file": "/submissions/titanic_submission_001.csv",
  "submission_id": "12345",
  "leaderboard_score": 0.79,
  "current_rank": "30th percentile"
}
```
**Cost:** Free

---

### 9. EvaluationAgent (LLM)
**Role:** Diagnoses model performance and identifies issues
**Input:**
- `training_results: Dict`
- `leaderboard_score: float`
- `execution_plan: Dict`

**Output:**
```python
{
  "cv_score": 0.823,
  "lb_score": 0.79,
  "gap": -0.033,
  "gap_type": "overfitting",

  "diagnosis": {
    "overfitting": true,
    "underfitting": false,
    "train_test_shift": "possible",
    "cv_reliable": true
  },

  "current_percentile": 0.30,
  "target_percentile": 0.20,
  "gap_to_target": 0.10,
  "improvement_needed": "+2-3% accuracy",

  "strengths": [
    "LightGBM strong (CV 0.815)",
    "Low fold variance (reliable)"
  ],

  "weaknesses": [
    "Overfitting by 3.3%",
    "Too many features?"
  ],

  "needs_improvement": true,  # ← DECISION FLAG

  "hypotheses": [
    "Add regularization (L1/L2)",
    "Drop low-importance features",
    "Simplify feature engineering"
  ]
}
```
**Cost:** 1 LLM call

---

### 10. StrategyOptimizer (LLM) - Conditional
**Role:** Suggests specific improvements and decides loop-back point
**Input:**
- `evaluation_diagnosis: Dict`
- `full_context: Dict` (all previous results)

**Output:**
```python
{
  "iteration": 2,
  "strategy_type": "refinement",  # vs "rebuild"

  "changes_recommended": {
    "preprocessing": "no change",
    "feature_engineering": {
      "action": "remove",
      "features_to_drop": ["title", "fare_per_person"],
      "reason": "Reduce overfitting"
    },
    "model_selection": {
      "models_to_try": ["lightgbm"],  # Only best
      "hyperparameter_changes": {
        "lightgbm": {
          "reg_alpha": 0.1,  # Add L1 reg
          "reg_lambda": 0.1,  # Add L2 reg
          "max_depth": 5      # Reduce from 7
        }
      }
    }
  },

  "loop_back_to": "feature_engineering",  # Phase 6

  "expected_improvement": {
    "cv_score": 0.81,
    "lb_score": 0.805,
    "percentile": 0.18
  },

  "confidence": "medium-high"
}
```
**Cost:** 1 LLM call (only if needs_improvement == true)

---

## 📁 Project Structure

```
src/
├── agents/
│   ├── base.py                          # BaseAgent for all workers
│   │
│   ├── llm_agents/                      # AI Decision Makers
│   │   ├── base_llm_agent.py            # Base for LLM agents
│   │   ├── problem_understanding_agent.py    # ✅ IMPLEMENTED
│   │   ├── data_analysis_agent.py       # ✅ IMPLEMENTED
│   │   ├── preprocessing_agent.py       # 🚧 TO IMPLEMENT (Day 5-6)
│   │   ├── planning_agent.py            # ✅ IMPLEMENTED
│   │   ├── feature_engineering_agent.py # 🚧 TO IMPLEMENT (Day 8-9)
│   │   ├── evaluation_agent.py          # 🚧 TO IMPLEMENT (Day 15-16)
│   │   └── strategy_agent.py            # 🚧 TO IMPLEMENT (Day 15-16)
│   │
│   ├── orchestrator/
│   │   ├── orchestrator.py              # ✅ NEEDS REFACTOR (Day 3-4)
│   │   └── phases.py                    # ✅ NEEDS REFACTOR (Day 3-4)
│   │
│   ├── data_collector/
│   │   └── collector.py                 # ✅ IMPLEMENTED
│   │
│   ├── model_trainer/
│   │   ├── trainer.py                   # ✅ NEEDS REFACTOR (Day 10-11)
│   │   ├── data_pipeline.py             # ✅ NEEDS REFACTOR (Day 10-11)
│   │   ├── detection.py                 # Task/model type detection
│   │   └── models/                      # Model implementations
│   │       ├── tabular/
│   │       │   ├── lightgbm.py          # ✅ IMPLEMENTED
│   │       │   ├── xgboost.py           # ✅ IMPLEMENTED
│   │       │   └── pytorch_mlp.py       # ✅ IMPLEMENTED
│   │       ├── nlp/
│   │       │   ├── transformer.py       # ✅ IMPLEMENTED
│   │       │   └── bert_classifier.py   # 🚧 TO IMPLEMENT
│   │       ├── vision/                  # 🔮 FUTURE (architecture ready)
│   │       │   ├── resnet.py
│   │       │   └── efficientnet.py
│   │       └── timeseries/              # 🔮 FUTURE (architecture ready)
│   │           ├── lstm.py
│   │           └── prophet.py
│   │
│   ├── submission/
│   │   └── submitter.py                 # ✅ IMPLEMENTED
│   │
│   └── leaderboard/
│       └── monitor.py                   # ✅ IMPLEMENTED
│
└── main.py                              # ✅ Entry point
```

---

## 🚀 Implementation Status

### ✅ Fully Implemented (Working Today)
- BaseAgent architecture
- Orchestrator workflow (needs refactoring for Option B)
- Data collection via Kaggle API
- Problem understanding agent
- Data analysis agent
- Planning agent
- Tabular model training (LightGBM, XGBoost, PyTorch MLP)
- Basic NLP support (transformers)
- Submission handling
- Leaderboard monitoring

### 🚧 To Implement (Days 3-22)
- **Day 3-4:** Fix orchestrator flow (remove duplicates, context passing)
- **Day 5-6:** PreprocessingAgent (code generation for tabular + NLP)
- **Day 7:** Test Phase 1-4 on Titanic
- **Day 8-9:** FeatureEngineeringAgent (code generation)
- **Day 10-11:** Refactor ModelTrainer/DataPipeline to use execution_plan
- **Day 12-13:** End-to-end testing (tabular + NLP)
- **Day 15-16:** EvaluationAgent + StrategyOptimizer
- **Day 17-18:** Test on 3rd competition
- **Day 19-20:** Logging and error handling
- **Day 21:** Final documentation

### 🔮 Future Work (Post-Nov 27)
- **Computer Vision:** ResNet, EfficientNet, ViT implementations
- **Time Series:** LSTM, Prophet, ARIMA implementations
- **Audio:** Speech recognition models
- **Multi-modal:** Combined approaches
- **Advanced Features:**
  - Parallel model training
  - Hyperparameter optimization (Optuna)
  - Advanced ensembling (stacking, blending)
  - External data collection
  - Meta-learning from past competitions

---

## 🔧 Key Implementation Principles

### 1. Modality Detection is Critical

DataAnalysisAgent MUST output `data_modality` accurately:
```python
{
  "data_modality": "tabular|nlp|vision|timeseries|audio|mixed"
}
```

This determines:
- Which preprocessing code to generate
- Which feature engineering to apply
- Which models to use

### 2. Conditional Agent Invocation

Orchestrator checks flags before calling agents:
```python
# Phase 4
if data_analysis["preprocessing_required"]:
    preprocessing_result = await preprocessing_agent.run(context)
    data_path = preprocessing_result["clean_data_path"]
else:
    logger.info("⏭️  Skipping preprocessing - data is clean")
    data_path = raw_data_path
```

### 3. Code Generation is Key

PreprocessingAgent and FeatureEngineeringAgent don't execute logic directly - they generate Python code that executors run. This allows:
- Full transparency (see exact code)
- Easy debugging (check generated code)
- Reproducibility (save code for later)
- Safety (review before execution)

### 4. Context Accumulation

Each agent adds to context:
```python
context = {
  "competition_name": "titanic",
  "problem_understanding": {...},   # From Phase 2
  "data_analysis": {...},            # From Phase 3
  "clean_data_path": "...",          # From Phase 4
  "execution_plan": {...},           # From Phase 5
  "featured_data_path": "...",       # From Phase 6
  "training_results": {...},         # From Phase 7
  "evaluation": {...}                # From Phase 9
}
```

Downstream agents receive full context and make informed decisions.

### 5. No Fallback Policy

If AI fails, system fails. No hardcoded fallbacks:
```python
if not execution_plan:
    raise RuntimeError(
        "❌ No execution plan from AI. "
        "This is a pure agentic system - requires AI. "
        "Check GEMINI_API_KEY."
    )
```

This ensures the system stays truly universal.

---

## 🎯 Success Metrics

**Primary Goal:** Achieve top 20% on tabular AND NLP competitions

**System Metrics:**
- Competition types successfully handled (target: 2+ by Nov 27)
- Average percentile ranking achieved
- Time to reach top 20%
- Cost per competition (LLM calls)

**Quality Metrics:**
- Zero hardcoded competition-specific logic
- Successful handling of different data modalities
- Strategy improvement across iterations

---

## 📋 22-Day Implementation Plan

### Week 1 (Nov 5-11): Foundation
- **Nov 5-6:** ✅ Architecture finalization & documentation
- **Nov 7-8:** Fix orchestrator (remove duplicates, context passing)
- **Nov 9-10:** Implement PreprocessingAgent
- **Nov 11:** Test Phases 1-4 on Titanic

### Week 2 (Nov 12-18): Core Implementation
- **Nov 12-13:** Implement FeatureEngineeringAgent
- **Nov 14-15:** Refactor ModelTrainer + DataPipeline
- **Nov 16-17:** End-to-end test (Titanic + NLP)
- **Nov 18:** Week 2 review

### Week 3 (Nov 19-25): Polish & Testing
- **Nov 19-20:** Iteration loop (Evaluation + Optimizer)
- **Nov 21-22:** Test on 3rd competition
- **Nov 23-24:** Logging and error handling
- **Nov 25:** Final documentation

### Week 4 (Nov 26-27): Buffer & Submission
- **Nov 26:** Buffer for unexpected issues
- **Nov 27:** Final testing and SUBMISSION 🎯

---

## 🔑 Environment Setup

### Required API Keys

```bash
# .env file
GEMINI_API_KEY=your-gemini-key          # For AI agents (FREE TIER OK)
KAGGLE_USERNAME=your-username            # For Kaggle API
KAGGLE_KEY=your-kaggle-key              # For Kaggle API
```

### Running the System

```bash
# Install dependencies
pip install -r requirements.txt

# Run for any competition
python src/main.py

# The system will:
# 1. Understand competition problem
# 2. Analyze data
# 3. Generate preprocessing code (if needed)
# 4. Create execution plan
# 5. Generate feature engineering code (if needed)
# 6. Train models
# 7. Submit and monitor
# 8. Iterate until top 20%
```

---

## 🌟 Extending to New Modalities (Future)

### Adding Vision Support (Example)

**Step 1:** Implement model
```python
# src/agents/model_trainer/models/vision/resnet.py
async def train_resnet(X, y, config, models_dir):
    """Train ResNet50 for image classification"""
    # Implementation here
    pass
```

**Step 2:** Update ModelTrainer routing
```python
# src/agents/model_trainer/trainer.py
elif modality == "vision":
    if "resnet" in model_name:
        return train_resnet(X, y, config, models_dir)
```

**Step 3:** Test on vision competition
```bash
python src/main.py --competition "digit-recognizer"
```

Architecture handles the rest automatically! ✨

---

## 💡 Remember

This documentation is the **north star** for all development. Every code change should move toward:
1. **More universal** - handles more competition types
2. **More intelligent** - AI makes more decisions
3. **Less hardcoded** - fewer assumptions in code
4. **More autonomous** - less human intervention needed

When in doubt, ask: "Will this work for a competition type we've never seen before?"

---

**Last Updated:** November 5, 2024
**Target Submission:** November 27, 2024
**Implementation Strategy:** Option B (Core Modalities)
**Status:** Architecture Finalized ✅ Ready for Implementation