"""
Data Analysis Agent
AI-powered agent that analyzes datasets and suggests preprocessing strategies.
"""
import logging
from typing import Dict, Any
from .base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)


class DataAnalysisAgent(BaseLLMAgent):
    """AI agent that intelligently analyzes data and suggests preprocessing."""

    def __init__(self):
        system_prompt = """You are an expert data scientist specializing in Kaggle competitions.

Your role is to analyze datasets and provide actionable preprocessing and feature engineering recommendations.

You excel at:
- Identifying data quality issues (missing values, outliers, imbalance)
- Detecting feature types (numerical, categorical, text, datetime)
- Suggesting appropriate preprocessing strategies
- Recommending feature engineering approaches
- Identifying the target variable
- Detecting data leakage risks

Provide specific, actionable recommendations that improve model performance."""

        super().__init__(
            name="DataAnalysisAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.3,  # Lower temperature for more focused analysis
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

        prompt = f"""Analyze this Kaggle competition dataset and provide recommendations.

## Competition: {competition_name}

## Dataset Information
{dataset_desc}

Please analyze and provide:

1. **Target Variable**: Which column is most likely the target? Why?
2. **Data Quality Issues**: Missing values, outliers, class imbalance, etc.
3. **Feature Types**: Categorize features (numerical, categorical, text, datetime, ID)
4. **Preprocessing Strategy**:
   - How to handle missing values?
   - How to encode categorical features?
   - Should we scale/normalize features?
   - Any feature transformations needed (log, binning, etc.)?
5. **Feature Engineering**: Top 3-5 new features to create
6. **Potential Issues**: Data leakage risks, correlated features, etc.
7. **Model Recommendations**: Which ML models suit this data?

Respond with JSON:
{{
    "target_column": "column_name",
    "target_confidence": "high|medium|low",
    "task_type": "binary_classification|multiclass|regression",
    "data_quality": {{
        "missing_values_strategy": "...",
        "outliers": "...",
        "class_balance": "balanced|imbalanced"
    }},
    "preprocessing": {{
        "categorical_encoding": "label|onehot|target",
        "numerical_scaling": "standard|minmax|robust|none",
        "handle_missing": "mean|median|mode|drop",
        "feature_transformations": ["log(feature_x)", "binning(age)"]
    }},
    "feature_engineering": [
        "interaction: feature_a * feature_b",
        "polynomial: feature_x^2",
        "aggregation: groupby mean"
    ],
    "warnings": ["potential_data_leakage_in_X", "high_cardinality_in_Y"],
    "recommended_models": ["lightgbm", "xgboost", "neural_network"],
    "confidence": "high|medium|low"
}}"""

        try:
            analysis = await self.reason_json(prompt, context)
            logger.info(f"AI Data Analysis complete for {competition_name}")
            return analysis

        except Exception as e:
            logger.error(f"Error in data analysis: {e}")
            return self._fallback_analysis(dataset_info)

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

    def _fallback_analysis(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis if AI fails."""
        logger.warning("Using fallback data analysis")

        return {
            "target_column": "unknown",
            "target_confidence": "low",
            "task_type": "classification",
            "data_quality": {
                "missing_values_strategy": "Use mean for numerical, mode for categorical",
                "outliers": "No specific strategy",
                "class_balance": "unknown"
            },
            "preprocessing": {
                "categorical_encoding": "label",
                "numerical_scaling": "standard",
                "handle_missing": "mean",
                "feature_transformations": []
            },
            "feature_engineering": [],
            "warnings": ["Fallback analysis due to AI error"],
            "recommended_models": ["lightgbm", "xgboost"],
            "confidence": "low"
        }