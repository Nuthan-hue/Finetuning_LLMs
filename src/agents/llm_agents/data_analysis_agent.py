"""
Data Analysis Agent
AI-powered agent that analyzes datasets and suggests preprocessing strategies.
"""
import logging
from typing import Dict, Any
from pathlib import Path
from .base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)


class DataAnalysisAgent(BaseLLMAgent):
    """AI agent that intelligently analyzes data and suggests preprocessing."""

    def __init__(self):
        # Load system prompt from file
        prompt_file = Path(__file__).parent.parent / "prompts" / "data_analysis_agent.txt"
        system_prompt = prompt_file.read_text()

        super().__init__(
            name="DataAnalysisAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.2,  # Very low temperature for precise analysis
            system_prompt=system_prompt
        )

    async def analyze_and_suggest(
        self,
        dataset_info: Dict[str, Any],
        competition_name: str
    ) -> Dict[str, Any]:
        """
        Analyze dataset and suggest preprocessing strategies.

        Args:
            dataset_info: Dictionary with dataset statistics
            competition_name: Name of competition

        Returns:
            Dictionary with analysis and recommendations
        """
        context = {
            "competition": competition_name,
            "num_files": len(dataset_info.get("datasets", {})),
            "total_rows": sum(d.get("rows", 0) for d in dataset_info.get("datasets", {}).values()),
            "features": dataset_info.get("datasets", {})
        }

        # Build detailed dataset description
        dataset_desc = self._format_dataset_info(dataset_info)

        prompt = f"""Analyze this Kaggle competition dataset and provide COMPLETE task understanding.

## Competition: {competition_name}

## Dataset Information
{dataset_desc}

Look for sample_submission.csv to understand the output format!

Provide detailed Kaggle-specific analysis in JSON format:

{{
    "task_type": "binary_classification|multiclass_classification|regression|clustering|time_series_forecasting|ranking|anomaly_detection|object_detection|segmentation|nlp_classification|nlp_generation",
    "data_modality": "tabular|nlp|computer_vision|time_series|audio|mixed",
    "has_target": true|false,
    "target_column": "column_name or null if unsupervised",
    "target_confidence": "high|medium|low",
    "target_characteristics": {{
        "type": "binary|multiclass|continuous|ordinal",
        "num_classes": 2,
        "classes": [0, 1],
        "distribution": {{"0": 0.62, "1": 0.38}},
        "is_imbalanced": true|false,
        "range": [min, max] // for regression
    }},
    "evaluation_metric": "accuracy|rmse|rmsle|f1|auc|mae|logloss|custom",
    "submission_format": {{
        "id_column": "PassengerId",
        "prediction_column": "Survived",
        "output_type": "integer|float|binary|probabilities|class_labels",
        "requires_transformation": "round|threshold|argmax|none"
    }},
    "data_quality": {{
        "missing_values": {{"column": percentage}},
        "outliers": ["column1", "column2"],
        "class_balance": "balanced|imbalanced",
        "issues": ["high_cardinality_in_X", "data_leakage_risk"]
    }},
    "preprocessing_required": true|false,
    "feature_types": {{
        "id_columns": ["PassengerId"],
        "numerical": ["Age", "Fare"],
        "categorical": ["Sex", "Embarked"],
        "text": ["Name"],
        "datetime": ["Date"],
        "target": ["Survived"]
    }},
    "preprocessing": {{
        "categorical_encoding": "label|onehot|target|embeddings",
        "numerical_scaling": "standard|minmax|robust|none",
        "handle_missing": "mean|median|mode|drop|knn",
        "text_processing": "tfidf|word2vec|bert|none",
        "feature_transformations": ["log(Fare)", "bin(Age,bins=5)"]
    }},
    "feature_engineering": [
        "family_size = SibSp + Parch + 1",
        "is_alone = (family_size == 1)",
        "title = extract from Name",
        "fare_per_person = Fare / family_size"
    ],
    "recommended_models": ["lightgbm", "xgboost", "catboost", "neural_network", "ensemble"],
    "model_justification": "LightGBM for speed, XGBoost for accuracy, ensemble for top 1%",
    "warnings": ["Potential data leakage in column X", "High missing rate in Age"],
    "confidence": "high|medium|low",
    "competition_strategy": "Focus on feature engineering. Target is imbalanced - use SMOTE or class weights."
}}

Be THOROUGH. Analyze sample_submission.csv format carefully!"""

        analysis = await self.reason_json(prompt, context)

        if not analysis or "target_column" not in analysis:
            raise RuntimeError(
                f"❌ AI Data Analysis failed for {competition_name}. "
                "Pure agentic AI system - no fallback! "
                "Ensure GEMINI_API_KEY is set and valid."
            )

        logger.info(f"🤖 AI Data Analysis complete for {competition_name}")
        logger.info(f"   Task: {analysis.get('task_type')}")
        logger.info(f"   Target: {analysis.get('target_column')}")
        logger.info(f"   Metric: {analysis.get('evaluation_metric')}")

        return analysis

    def _format_dataset_info(self, dataset_info: Dict[str, Any]) -> str:
        """Format dataset info for prompt."""
        lines = []

        datasets = dataset_info.get("datasets", {})
        for filename, info in datasets.items():
            lines.append(f"\n### {filename}")
            lines.append(f"- Rows: {info.get('rows', 'N/A')}")
            lines.append(f"- Columns: {info.get('columns', 'N/A')}")

            if "column_types" in info:
                lines.append(f"- Column Types:")
                for col_type, cols in info["column_types"].items():
                    if cols:
                        lines.append(f"  - {col_type}: {', '.join(cols[:5])}")

            if "missing_values" in info:
                missing = info["missing_values"]
                if missing:
                    lines.append(f"- Missing Values: {len(missing)} columns affected")

            if "dtypes" in info:
                dtypes_summary = {}
                for dtype in info["dtypes"].values():
                    dtypes_summary[dtype] = dtypes_summary.get(dtype, 0) + 1
                lines.append(f"- Data Types: {dtypes_summary}")

        return "\n".join(lines)
