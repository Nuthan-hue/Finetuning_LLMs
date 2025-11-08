"""
Planning Agent
AI-powered agent that creates comprehensive execution plans based on problem understanding and data analysis.
"""
import logging
import json
from pathlib import Path
from typing import Dict, Any
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class PlanningAgent(BaseLLMAgent):
    """
    AI agent that creates detailed execution plans for Kaggle competitions.

    This agent combines problem understanding and data analysis to create
    a comprehensive, step-by-step plan that workers can execute.
    """

    def __init__(self):
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "planning_agent.txt"
        system_prompt = prompt_file.read_text()

        super().__init__(
            name="PlanningAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.3,
            system_prompt=system_prompt
        )

    async def create_plan(
        self,
        problem_understanding: Dict[str, Any],
        dataset_info: Dict[str, Any],
        competition_name: str
    ) -> Dict[str, Any]:
        """
        Create comprehensive execution plan based on problem understanding and raw dataset info.

        This method does BOTH data analysis AND planning in one comprehensive step.

        Args:
            problem_understanding: Output from ProblemUnderstandingAgent
            dataset_info: Raw dataset information (from DataCollector analysis_report)
            competition_name: Name of the competition

        Returns:
            Dictionary containing detailed execution plan with embedded data analysis:
            {
                "competition_name": str,
                "models_to_try": [
                    {
                        "model": "lightgbm|xgboost|pytorch_mlp|transformer|...",
                        "priority": 1,
                        "reason": "Why this model is suitable",
                        "hyperparameters": {...},
                        "expected_performance": "Performance expectation"
                    },
                    ...
                ],
                "preprocessing_plan": [
                    {
                        "step": "remove_ids",
                        "details": {...},
                        "reason": "..."
                    },
                    ...
                ],
                "feature_engineering_plan": [
                    {
                        "feature_name": "...",
                        "formula": "...",
                        "reason": "...",
                        "priority": 1
                    },
                    ...
                ],
                "validation_strategy": {
                    "method": "kfold|stratified_kfold|timeseries_split|...",
                    "n_splits": 5,
                    "stratify_column": "...",
                    "shuffle": true,
                    "reason": "..."
                },
                "hyperparameter_tuning": {
                    "method": "grid_search|random_search|bayesian|optuna|...",
                    "n_trials": 100,
                    "time_budget_minutes": 60,
                    "key_parameters": {...}
                },
                "ensemble_strategy": {
                    "use_ensemble": true,
                    "method": "voting|stacking|blending|weighted_average|...",
                    "models_to_combine": [...],
                    "weights": {...}
                },
                "success_criteria": {
                    "target_metric_value": 0.85,
                    "stop_if_exceeded": true,
                    "max_training_time_hours": 2
                },
                "execution_order": [
                    "Step 1: Preprocess data following preprocessing_plan",
                    "Step 2: Engineer features following feature_engineering_plan",
                    ...
                ]
            }
        """
        logger.info(f"🧠 Creating comprehensive plan (data analysis + execution) for: {competition_name}")

        # Build comprehensive planning prompt (includes data analysis)
        prompt = self._build_planning_prompt(
            problem_understanding,
            dataset_info,
            competition_name
        )

        try:
            # Get AI planning (does BOTH analysis and planning)
            response_text = generate_ai_response(self.model, prompt)

            # Parse JSON response
            plan = self._parse_ai_response(response_text)

            # Add metadata
            plan["competition_name"] = competition_name
            plan["ai_model"] = self.model_name
            plan["based_on"] = {
                "problem_understanding": problem_understanding.get("task_type"),
                "problem_type": problem_understanding.get("competition_type")
            }

            logger.info(f"✅ Comprehensive plan created!")
            logger.info(f"   Target: {plan.get('data_analysis', {}).get('target_column', 'TBD')}")
            logger.info(f"   Models: {len(plan['models_to_try'])}")
            logger.info(f"   Features: {len(plan['feature_engineering_plan'])}")
            logger.info(f"   Preprocessing steps: {len(plan['preprocessing_plan'])}")

            return plan

        except Exception as e:
            logger.error(f"AI planning failed: {e}")
            raise RuntimeError(
                f"❌ AI failed to create execution plan: {str(e)}\n"
                "This is a pure agentic AI system - no fallback logic available."
            )

    def _build_planning_prompt(
        self,
        problem_understanding: Dict[str, Any],
        data_analysis: Dict[str, Any],
        competition_name: str
    ) -> str:
        """
        Build comprehensive prompt for AI to create execution plan.

        Args:
            problem_understanding: Problem understanding from AI
            data_analysis: Data analysis from AI
            competition_name: Competition name

        Returns:
            Formatted prompt string
        """
        return f"""You are an expert Kaggle competition strategist. Create a comprehensive, executable plan for winning this competition.

# Competition Context

**Competition:** {competition_name}

## Problem Understanding
{json.dumps(problem_understanding, indent=2)}

## Data Analysis
{json.dumps(data_analysis, indent=2)}

# Your Task

Create a DETAILED, EXECUTABLE plan that workers can follow step-by-step. This plan must be:
1. **Specific** - Exact steps, not vague suggestions
2. **Prioritized** - Most important/effective actions first
3. **Complete** - Cover all aspects: preprocessing, features, models, validation
4. **Realistic** - Achievable with available tools and time

# Plan Components

## 1. Models to Try
- List 3-5 models in PRIORITY order (best first)
- For each model:
  - Model type (lightgbm, xgboost, pytorch_mlp, transformer, etc.)
  - Why it's suitable for THIS specific problem
  - Specific hyperparameters to start with
  - Expected performance level

## 2. Preprocessing Plan
Based on the data analysis, specify EXACT preprocessing steps:
- Which columns to remove (IDs, etc.)
- How to handle missing values (column by column if needed)
- Outlier detection and treatment
- Encoding strategy (one-hot, label, target encoding)
- Scaling strategy (standard, minmax, robust, none)

## 3. Feature Engineering Plan
List SPECIFIC features to create:
- Feature name
- Exact formula/calculation
- Why it will help for THIS problem
- Priority (1=must have, 2=nice to have, 3=experimental)

## 4. Validation Strategy
- Cross-validation method (k-fold, stratified, time-series, etc.)
- Number of splits
- Whether to stratify (and on what column)
- Why this strategy fits the problem

## 5. Hyperparameter Tuning
- Method (grid search, random search, Bayesian, Optuna)
- Key parameters to tune for each model
- Time budget
- Number of trials

## 6. Ensemble Strategy
- Should we ensemble?
- Which models to combine?
- How to combine them (voting, stacking, weighted average)
- Weights or stacking meta-learner

## 7. Success Criteria
- Target metric value (based on competition metric)
- When to stop training
- Maximum time budget

## 8. Execution Order
Clear step-by-step execution sequence

# Output Format

Respond with ONLY a valid JSON object (no markdown, no code blocks):

{{
    "models_to_try": [
        {{
            "model": "lightgbm|xgboost|pytorch_mlp|transformer|...",
            "priority": 1,
            "reason": "Specific reason for THIS competition",
            "hyperparameters": {{"param1": "value1", ...}},
            "expected_performance": "e.g., 0.80-0.85 accuracy"
        }}
    ],
    "preprocessing_plan": [
        {{
            "step": "remove_ids|handle_missing|detect_outliers|encode_categorical|scale_numerical|...",
            "details": {{"specific": "implementation details"}},
            "reason": "Why this step is important",
            "order": 1
        }}
    ],
    "feature_engineering_plan": [
        {{
            "feature_name": "descriptive_name",
            "formula": "exact calculation",
            "reason": "why this helps",
            "priority": 1
        }}
    ],
    "validation_strategy": {{
        "method": "kfold|stratified_kfold|timeseries_split|train_test_split|...",
        "n_splits": 5,
        "stratify_column": "target_column_name or null",
        "shuffle": true,
        "random_state": 42,
        "reason": "why this validation strategy"
    }},
    "hyperparameter_tuning": {{
        "method": "optuna|grid_search|random_search|bayesian|...",
        "n_trials": 100,
        "time_budget_minutes": 60,
        "key_parameters": {{
            "lightgbm": ["learning_rate", "num_leaves", "max_depth"],
            "xgboost": ["learning_rate", "max_depth", "n_estimators"]
        }},
        "optimization_metric": "metric_to_optimize"
    }},
    "ensemble_strategy": {{
        "use_ensemble": true,
        "method": "voting|stacking|blending|weighted_average|...",
        "models_to_combine": ["model1", "model2", ...],
        "weights": {{"model1": 0.6, "model2": 0.4}} or null,
        "meta_learner": "logistic_regression|..." or null,
        "reason": "why ensemble will help"
    }},
    "success_criteria": {{
        "target_metric_value": 0.85,
        "target_percentile": 0.20,
        "stop_if_exceeded": true,
        "max_training_time_hours": 2,
        "early_stopping_patience": 50
    }},
    "execution_order": [
        "Step 1: Download and explore data",
        "Step 2: Execute preprocessing plan",
        "Step 3: Execute feature engineering plan",
        "Step 4: Train priority 1 model with cross-validation",
        "Step 5: Tune hyperparameters",
        "Step 6: Train remaining models",
        "Step 7: Create ensemble if beneficial",
        "Step 8: Generate and submit predictions"
    ],
    "risk_mitigation": {{
        "overfitting": "Use cross-validation, early stopping, regularization",
        "data_leakage": "Ensure train/test split integrity, no target leakage in features",
        "time_constraints": "Start with simplest effective model, then iterate"
    }},
    "confidence": "high|medium|low"
}}

Create the plan:"""

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse AI response into structured plan format.

        Args:
            response_text: Raw response from AI

        Returns:
            Parsed plan dictionary
        """
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Parse JSON
            plan = json.loads(cleaned)

            # Validate required fields
            required_fields = [
                "models_to_try",
                "preprocessing_plan",
                "feature_engineering_plan",
                "validation_strategy",
                "execution_order"
            ]

            for field in required_fields:
                if field not in plan:
                    raise ValueError(f"Missing required field in plan: {field}")

            logger.info(f"✅ AI plan parsed successfully")
            logger.info(f"   Models: {len(plan['models_to_try'])}")
            logger.info(f"   Features: {len(plan['feature_engineering_plan'])}")
            logger.info(f"   Preprocessing steps: {len(plan['preprocessing_plan'])}")

            return plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response: {response_text[:500]}")
            raise RuntimeError(
                "AI returned invalid JSON response for plan. "
                "This is a pure agentic AI system - no fallback parsing available."
            )
        except Exception as e:
            logger.error(f"Error parsing AI plan: {e}")
            raise

    def get_plan_summary(self, plan: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of execution plan.

        Args:
            plan: Execution plan dictionary

        Returns:
            Formatted summary string
        """
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║  EXECUTION PLAN: {plan['competition_name']:^43} ║
╚══════════════════════════════════════════════════════════════╝

🎯 MODELS TO TRY ({len(plan['models_to_try'])} models)
"""
        for i, model in enumerate(plan['models_to_try'], 1):
            summary += f"  {i}. {model['model'].upper()} (Priority {model['priority']})\n"
            summary += f"     Reason: {model['reason']}\n"
            summary += f"     Expected: {model.get('expected_performance', 'TBD')}\n"

        summary += f"""
🔧 PREPROCESSING PLAN ({len(plan['preprocessing_plan'])} steps)
"""
        for i, step in enumerate(sorted(plan['preprocessing_plan'], key=lambda x: x.get('order', 99)), 1):
            summary += f"  {i}. {step['step'].upper()}\n"
            summary += f"     → {step['reason']}\n"

        summary += f"""
💡 FEATURE ENGINEERING ({len(plan['feature_engineering_plan'])} features)
"""
        priority_features = [f for f in plan['feature_engineering_plan'] if f.get('priority', 3) == 1]
        for i, feat in enumerate(priority_features[:5], 1):  # Show top 5
            summary += f"  {i}. {feat['feature_name']}\n"
            summary += f"     Formula: {feat['formula']}\n"

        if len(priority_features) > 5:
            summary += f"  ... and {len(priority_features) - 5} more priority features\n"

        val_strategy = plan['validation_strategy']
        summary += f"""
✅ VALIDATION STRATEGY
  Method: {val_strategy['method']}
  Splits: {val_strategy.get('n_splits', 'N/A')}
  Stratify: {val_strategy.get('stratify_column', 'No')}
  Reason: {val_strategy.get('reason', 'TBD')}

🎛️  HYPERPARAMETER TUNING
  Method: {plan.get('hyperparameter_tuning', {}).get('method', 'None')}
  Trials: {plan.get('hyperparameter_tuning', {}).get('n_trials', 'N/A')}
  Budget: {plan.get('hyperparameter_tuning', {}).get('time_budget_minutes', 'N/A')} minutes

🤝 ENSEMBLE STRATEGY
  Use Ensemble: {plan.get('ensemble_strategy', {}).get('use_ensemble', False)}
  Method: {plan.get('ensemble_strategy', {}).get('method', 'N/A')}

🎯 SUCCESS CRITERIA
  Target Metric: {plan.get('success_criteria', {}).get('target_metric_value', 'TBD')}
  Target Percentile: Top {plan.get('success_criteria', {}).get('target_percentile', 0.20) * 100}%
  Max Time: {plan.get('success_criteria', {}).get('max_training_time_hours', 'N/A')} hours

📋 EXECUTION ORDER
"""
        for step in plan['execution_order']:
            summary += f"  → {step}\n"

        summary += f"""
🔍 Confidence: {plan.get('confidence', 'unknown').upper()}
"""
        return summary