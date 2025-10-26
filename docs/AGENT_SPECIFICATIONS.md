# AGENT SPECIFICATIONS
## Detailed Technical Specifications for All Agents

---

## Table of Contents
1. [Phase 0: Reasoning Agents](#phase-0-reasoning-agents)
2. [Phase 1: Data Collection](#phase-1-data-collection)
3. [Phase 2: Model Training](#phase-2-model-training)
4. [Phase 3: Submission](#phase-3-submission)
5. [Phase 4: Learning & Monitoring](#phase-4-learning--monitoring)
6. [Agent Communication Protocol](#agent-communication-protocol)

---

## Phase 0: Reasoning Agents

### 1. CompetitionIntelligenceAgent

**Role:** Information Gatherer - Collects all available competition data

**Inputs:**
```python
{
    "competition_name": str  # e.g., "titanic"
}
```

**Processing Steps:**
1. Execute `kaggle competitions list -s {competition_name}`
2. Execute `kaggle competitions files -c {competition_name}`
3. Scrape competition overview page using BeautifulSoup
4. Download sample_submission.csv if available
5. Parse evaluation metric from competition page
6. Extract rules and timeline information
7. Check for discussion forum (for later mining)

**Outputs:**
```python
{
    "competition_name": str,
    "title": str,
    "url": str,
    "description": str,  # Full HTML/text description
    "evaluation_metric": str,  # e.g., "AUC", "RMSE", "F1"
    "metric_description": str,  # How metric is calculated
    "submission_format": {
        "file_type": str,  # "csv", "json", etc.
        "columns": [str],  # Required column names
        "sample_row": dict  # Example row from sample_submission
    },
    "rules": {
        "external_data_allowed": bool,
        "team_size_limit": int,
        "daily_submission_limit": int,
        "max_ensemble_size": int,  # if specified
        "special_requirements": [str]
    },
    "timeline": {
        "start_date": str,
        "end_date": str,
        "days_remaining": int
    },
    "data_files": [
        {
            "name": str,  # "train.csv"
            "size": int,  # bytes
            "columns": [str]  # if CSV
        }
    ],
    "prize": str,  # "$100,000"
    "participants": int,
    "teams": int
}
```

**Error Handling:**
- If competition not found: raise `CompetitionNotFoundError`
- If Kaggle API fails: retry 3 times with exponential backoff
- If scraping fails: return partial data with warnings

**Dependencies:**
- Kaggle CLI
- BeautifulSoup4
- requests
- pandas (for CSV parsing)

---

### 2. ProblemReasoningAgent ⭐

**Role:** The Brain - Deeply understands the competition problem

**Inputs:**
```python
{
    "competition_intelligence": dict,  # From CompetitionIntelligenceAgent
    "data_sample": pd.DataFrame  # First 100 rows of train data
}
```

**Processing Steps:**

1. **Initial Analysis:**
   ```python
   # Analyze data characteristics
   - Number of rows/columns
   - Data types (numeric, categorical, text, binary)
   - Missing value patterns
   - Target variable distribution
   - Feature value ranges
   ```

2. **LLM-Based Reasoning:**
   ```
   PROMPT TO LLM:

   You are an expert data scientist analyzing a Kaggle competition.

   Competition: {title}
   Description: {description}

   Data characteristics:
   - Rows: {n_rows}, Columns: {n_cols}
   - Column types: {type_summary}
   - Sample data:
   {data_sample.head()}

   Task: Analyze this competition and answer:

   1. What is the core task? (classification, regression, generation, ranking, etc.)

   2. What data modalities are involved?
      - Tabular data? (numeric/categorical features)
      - Text data? (identify text columns and characteristics)
      - Image data? (file paths to images?)
      - Time series? (temporal dependencies?)
      - Graph/relational data?
      - Multi-modal? (combination)

   3. What is unique or challenging about this problem?

   4. What domain knowledge is relevant?
      (healthcare, finance, NLP, computer vision, etc.)

   5. Based on the evaluation metric "{metric}", what should we optimize for?

   6. Are there any special requirements or constraints?

   Provide a comprehensive analysis in JSON format.
   ```

3. **Similarity Search:**
   - Search Kaggle for competitions with similar characteristics
   - Find competitions with same metric
   - Find competitions in same domain

4. **Synthesis:**
   - Combine LLM reasoning with similarity search
   - Generate problem understanding document

**Outputs:**
```python
{
    "problem_summary": str,  # 2-3 sentence clear summary

    "data_modalities": [
        {
            "type": str,  # "tabular", "text", "image", "time_series", "graph"
            "description": str,
            "columns": [str],  # Which columns contain this modality
            "characteristics": dict  # Modality-specific details
        }
    ],

    "task_category": str,  # "binary_classification", "regression", etc.

    "domain": {
        "primary": str,  # "Healthcare", "Finance", etc.
        "subdomain": str,  # "Disease Prediction", "Stock Forecasting"
        "relevant_expertise": [str]  # Domain knowledge needed
    },

    "evaluation_metric": {
        "name": str,  # "AUC", "RMSE"
        "optimization_direction": str,  # "maximize", "minimize"
        "interpretation": str,  # What does this metric measure?
        "typical_good_score": float,  # Based on similar competitions
        "custom_implementation_needed": bool
    },

    "key_challenges": [
        {
            "challenge": str,
            "severity": str,  # "high", "medium", "low"
            "potential_impact": str
        }
    ],

    "special_requirements": [str],

    "similar_competitions": [
        {
            "name": str,
            "url": str,
            "similarity_score": float,  # 0-1
            "similarity_reasons": [str],
            "winning_approaches": [str]  # If available
        }
    ],

    "confidence": float,  # 0-1: How confident is this analysis

    "reasoning_trace": str  # Full LLM output for debugging
}
```

**LLM Configuration:**
- Model: Gemini 1.5 Pro or GPT-4
- Temperature: 0.3 (balanced creativity/accuracy)
- Max tokens: 2000

---

### 3. SolutionStrategyAgent ⭐

**Role:** The Strategist - Researches and proposes winning approaches

**Inputs:**
```python
{
    "problem_understanding": dict,  # From ProblemReasoningAgent
    "competition_intelligence": dict
}
```

**Processing Steps:**

1. **Research Winning Solutions:**
   ```python
   # For each similar competition:
   - Search Kaggle discussion forums for "1st place solution"
   - Parse winning approach descriptions
   - Extract techniques, models, features used
   ```

2. **Search Recent Papers (Optional):**
   ```python
   # If relevant keywords identified:
   - Query ArXiv for recent papers
   - Focus on applied ML papers
   - Extract novel techniques
   ```

3. **Generate Solution Candidates:**
   ```
   PROMPT TO LLM:

   You are a Kaggle Grandmaster planning a competition strategy.

   Problem: {problem_summary}
   Data: {data_characteristics}
   Metric: {evaluation_metric}

   Based on these similar competitions:
   {similar_competitions_summary}

   Where winning approaches included:
   {winning_techniques}

   Task: Propose 3-5 solution strategies ranked by likelihood of success.

   For each strategy, specify:
   1. Overall approach (model type, technique)
   2. Detailed implementation steps
   3. Expected performance (based on similar competitions)
   4. Implementation complexity (Low/Medium/High)
   5. Estimated time to implement
   6. Key risks and mitigation
   7. References to similar successful applications

   Consider:
   - Traditional ML (XGBoost, LightGBM, CatBoost)
   - Deep Learning (Transformers, CNNs, RNNs, TabNet)
   - Ensemble methods
   - Feature engineering approaches
   - Novel techniques from recent research

   Provide detailed strategies in JSON format.
   ```

4. **Rank Strategies:**
   - Score based on: expected performance, feasibility, time
   - Consider available resources (GPU, time, API limits)

**Outputs:**
```python
{
    "recommended_strategies": [
        {
            "rank": int,  # 1 = top recommendation

            "name": str,  # "Gradient Boosting Ensemble"

            "approach_summary": str,  # One paragraph overview

            "expected_performance": {
                "percentile": float,  # Expected leaderboard percentile
                "metric_value": float,  # Expected metric value
                "confidence": float,  # 0-1
                "rationale": str  # Why this performance expected
            },

            "implementation": {
                "model_types": [str],  # ["LightGBM", "XGBoost"]
                "feature_engineering": [str],  # Specific features to create
                "preprocessing": [str],  # Steps needed
                "hyperparameter_strategy": str,  # "Optuna tuning", "Grid search"
                "validation_strategy": str,  # "5-fold CV", "Time-based split"
                "ensemble_method": str  # If applicable
            },

            "complexity": {
                "level": str,  # "Low", "Medium", "High"
                "estimated_time": str,  # "2-3 hours"
                "required_skills": [str],
                "dependencies": [str]  # Required libraries
            },

            "risks": [
                {
                    "risk": str,
                    "mitigation": str
                }
            ],

            "references": [
                {
                    "type": str,  # "kaggle_discussion", "paper", "blog"
                    "url": str,
                    "description": str
                }
            ]
        }
    ],

    "ensemble_recommendations": {
        "should_ensemble": bool,
        "method": str,  # "weighted_average", "stacking", "blending"
        "models_to_combine": [str]
    },

    "fallback_strategies": [
        # Simpler strategies if main ones fail
    ],

    "research_summary": str  # Summary of research findings
}
```

**Research Sources:**
1. Kaggle discussion forums (primary)
2. Kaggle winning solution notebooks
3. ArXiv papers (optional, for novel problems)
4. Blog posts from Kaggle Grandmasters

---

### 4. ImplementationPlannerAgent

**Role:** The Architect - Creates detailed execution plan

**Inputs:**
```python
{
    "problem_understanding": dict,
    "selected_strategy": dict,  # Top strategy from SolutionStrategyAgent
    "data_characteristics": dict
}
```

**Processing Steps:**

1. **Design Data Pipeline:**
   ```python
   # Based on data characteristics and strategy:
   - Define preprocessing steps (order matters!)
   - Specify feature engineering code
   - Design validation split strategy
   - Plan for data leakage prevention
   ```

2. **Configure Model:**
   ```python
   # Translate strategy to concrete configuration:
   - Specify model hyperparameters
   - Define training configuration
   - Set up experiment tracking
   ```

3. **Plan Experiments:**
   ```python
   # Define what to try:
   - Baseline experiment
   - Feature ablation experiments
   - Hyperparameter tuning experiments
   ```

**Outputs:**
```python
{
    "pipeline_specification": {
        "preprocessing": [
            {
                "step": str,  # "handle_missing_values"
                "method": str,  # "median_imputation"
                "parameters": dict,
                "applies_to": [str],  # Column names
                "order": int  # Execution order
            }
        ],

        "feature_engineering": [
            {
                "feature_name": str,
                "formula": str,  # Python code or description
                "rationale": str,  # Why this feature helps
                "expected_importance": str  # "high", "medium", "low"
            }
        ],

        "validation_strategy": {
            "method": str,  # "stratified_kfold"
            "n_splits": int,
            "shuffle": bool,
            "random_state": int,
            "stratify_on": str,  # Column name for stratification
            "rationale": str  # Why this validation strategy
        }
    },

    "model_configuration": {
        "model_type": str,  # "LightGBM"

        "hyperparameters": {
            # Model-specific hyperparameters
            "learning_rate": float,
            "n_estimators": int,
            "max_depth": int,
            # ... etc
        },

        "training_config": {
            "early_stopping_rounds": int,
            "verbose": int,
            "eval_metric": str,
            "categorical_features": [str],
            "use_gpu": bool
        },

        "tuning_strategy": {
            "method": str,  # "optuna", "grid_search", "random_search"
            "param_space": dict,
            "n_trials": int,
            "optimization_direction": str  # "maximize", "minimize"
        }
    },

    "experiment_plan": [
        {
            "experiment_name": str,
            "description": str,
            "what_to_try": str,
            "success_criteria": str,
            "estimated_time": str
        }
    ],

    "evaluation_specification": {
        "primary_metric": str,
        "secondary_metrics": [str],
        "logging_frequency": int,
        "what_to_log": [str]  # "feature_importance", "cv_scores", etc.
    },

    "implementation_steps": [
        {
            "step_number": int,
            "description": str,
            "code_sketch": str,  # Pseudocode
            "expected_output": str,
            "validation": str  # How to verify this step worked
        }
    ],

    "computational_requirements": {
        "estimated_memory": str,  # "16GB RAM"
        "estimated_time": str,  # "30 minutes"
        "gpu_required": bool,
        "disk_space": str  # "5GB"
    }
}
```

---

### 5. RiskAnalysisAgent

**Role:** The Validator - Identifies risks and ensures compliance

**Inputs:**
```python
{
    "problem_understanding": dict,
    "implementation_plan": dict,
    "competition_rules": dict
}
```

**Processing Steps:**

1. **Data Leakage Detection:**
   ```python
   # Check for:
   - Features that contain future information
   - Target variable leaking into features
   - Train/test contamination risks
   - Temporal leakage in time series
   ```

2. **Overfitting Risk Assessment:**
   ```python
   # Evaluate:
   - Model complexity vs dataset size
   - Validation strategy robustness
   - Regularization adequacy
   ```

3. **Competition Compliance Check:**
   ```python
   # Verify:
   - External data usage complies with rules
   - Submission format matches requirements
   - Team size within limits
   - No prohibited techniques
   ```

4. **Technical Risk Assessment:**
   ```python
   # Identify:
   - Resource constraints (memory, GPU, time)
   - Dependency issues
   - Reproducibility concerns
   ```

**Outputs:**
```python
{
    "risk_assessment": {
        "overall_risk_level": str,  # "low", "medium", "high"

        "risks": [
            {
                "category": str,  # "data_leakage", "overfitting", "compliance", "technical"
                "severity": str,  # "critical", "high", "medium", "low"
                "description": str,
                "likelihood": str,  # "high", "medium", "low"
                "impact": str,  # If this risk occurs, what happens?
                "mitigation": str,  # How to prevent or reduce this risk
                "status": str  # "mitigated", "accepted", "needs_attention"
            }
        ]
    },

    "compliance_check": {
        "status": str,  # "compliant", "non_compliant", "unclear"

        "checks": {
            "external_data": {
                "allowed": bool,
                "plan_uses_external_data": bool,
                "compliant": bool,
                "notes": str
            },
            "submission_format": {
                "validated": bool,
                "issues": [str]
            },
            "team_size": {
                "limit": int,
                "current": int,
                "compliant": bool
            },
            "prohibited_techniques": {
                "any_used": bool,
                "details": [str]
            }
        }
    },

    "validation_recommendations": [
        {
            "recommendation": str,
            "rationale": str,
            "priority": str  # "must_do", "should_do", "nice_to_have"
        }
    ],

    "monitoring_plan": {
        "what_to_monitor": [str],  # Metrics to watch during training
        "red_flags": [str],  # Warning signs to watch for
        "checkpoints": [str]  # When to validate assumptions
    }
}
```

---

## PROBLEM_CONTEXT Schema

**The aggregated output of Phase 0:**

```python
PROBLEM_CONTEXT = {
    "competition_name": str,

    "intelligence": {
        # Full output from CompetitionIntelligenceAgent
    },

    "understanding": {
        # Full output from ProblemReasoningAgent
    },

    "strategy": {
        # Full output from SolutionStrategyAgent
        # Including all ranked strategies (not just top one)
    },

    "implementation": {
        # Full output from ImplementationPlannerAgent
    },

    "risks": {
        # Full output from RiskAnalysisAgent
    },

    "metadata": {
        "phase_0_completed_at": str,  # ISO timestamp
        "phase_0_duration": float,  # seconds
        "llm_calls_made": int,
        "total_tokens_used": int,
        "estimated_api_cost": float
    }
}
```

---

## Phase 1-4 Agents (Brief Specs)

### DataCollectorAgent (Enhanced)

**New Inputs:**
- `PROBLEM_CONTEXT`

**Enhanced Behavior:**
- Uses `understanding.data_modalities` to validate downloaded data
- Checks for issues mentioned in `risks`
- Collects external data if `strategy` requires it

---

### ModelTrainerAgent (Enhanced)

**New Inputs:**
- `PROBLEM_CONTEXT.implementation`
- `PROBLEM_CONTEXT.risks`

**Enhanced Behavior:**
- Implements exact pipeline from `implementation.pipeline_specification`
- Uses exact model config from `implementation.model_configuration`
- Monitors for red flags from `risks.monitoring_plan`
- Logs metrics specified in `implementation.evaluation_specification`

---

### SubmissionAgent (Enhanced)

**New Inputs:**
- `PROBLEM_CONTEXT.intelligence.submission_format`

**Enhanced Behavior:**
- Validates output matches required format exactly
- Applies any post-processing from implementation plan

---

### LeaderboardMonitorAgent (Existing - No Changes)

---

### PerformanceAnalysisAgent (NEW!)

**Inputs:**
```python
{
    "leaderboard_score": float,
    "local_cv_scores": [float],
    "training_logs": dict,
    "problem_context": dict
}
```

**Outputs:**
```python
{
    "diagnosis": str,
    "score_analysis": {
        "lb_score": float,
        "cv_mean": float,
        "cv_std": float,
        "gap": float,  # LB - CV
        "interpretation": str
    },
    "root_causes": [str],
    "improvement_opportunities": [
        {
            "action": str,
            "expected_impact": str,
            "effort": str,
            "priority": int
        }
    ]
}
```

---

### StrategyRefinementAgent (NEW!)

**Inputs:**
```python
{
    "performance_diagnosis": dict,
    "current_strategy": dict,
    "problem_context": dict,
    "leaderboard_percentile": float,
    "iterations_so_far": int
}
```

**Outputs:**
```python
{
    "decision": str,  # "refine_current", "try_next_strategy", "ensemble", "done"
    "updated_plan": dict,  # Modified implementation plan
    "rationale": str,
    "expected_improvement": str,
    "next_action": str
}
```

---

## Agent Communication Protocol

### Message Format
All agents communicate via structured dictionaries with:
- Input validation
- Type checking
- Error propagation

### Error Handling
```python
{
    "success": bool,
    "data": dict,  # If success=True
    "error": {  # If success=False
        "type": str,
        "message": str,
        "traceback": str,
        "agent": str
    }
}
```

### State Management
Each agent maintains:
```python
{
    "state": "IDLE" | "RUNNING" | "COMPLETED" | "ERROR",
    "progress": float,  # 0-1
    "last_updated": str,  # ISO timestamp
    "results": dict
}
```

---

**END OF AGENT SPECIFICATIONS**