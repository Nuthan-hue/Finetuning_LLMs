"""
Data Analysis Agent
AI-powered agent that performs comprehensive statistical analysis and suggests preprocessing strategies.
"""
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class DataAnalysisAgent(BaseLLMAgent):
    """AI agent that analyzes data statistically and provides structured recommendations."""

    def __init__(self):
        # Load system prompt from file
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "data_analysis_agent.txt"
        system_prompt = prompt_file.read_text()

        super().__init__(
            name="DataAnalysisAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.2,  # Low temperature for precise analysis
            system_prompt=system_prompt
        )

    async def analyze_and_suggest(
        self,
        data_path: str,
        competition_name: str,
        problem_understanding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis and provide preprocessing recommendations.

        Args:
            data_path: Path to directory containing train.csv and test.csv
            competition_name: Name of the competition
            problem_understanding: Output from ProblemUnderstandingAgent

        Returns:
            Dictionary with analysis results and preprocessing recommendations
        """
        logger.info(f"🔍 Analyzing data for {competition_name}...")

        # Step 1: Load data and compute statistics
        statistics = self._compute_statistics(data_path)

        logger.info(f"✅ Computed statistics for train: {statistics['train_shape']}, test: {statistics['test_shape']}")

        # Step 2: Build prompt with statistics
        prompt = self._build_analysis_prompt(statistics, problem_understanding, competition_name)

        # Step 3: Get AI recommendations
        logger.info("🤖 Requesting AI analysis and recommendations...")
        response = generate_ai_response(self.model, prompt)

        # Step 4: Parse JSON response
        try:
            analysis = self._parse_json_response(response)
            logger.info(f"✅ Analysis complete: modality={analysis.get('data_modality')}, preprocessing_required={analysis.get('preprocessing_required')}")

            # Add file mapping from statistics (NOT from AI - this is internal info)
            analysis["data_files"] = statistics["data_files"]

            return analysis

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI response as JSON: {e}")
            logger.debug(f"Raw response: {response[:500]}...")
            raise RuntimeError("AI did not return valid JSON. Check prompt or model response.")

    def _compute_statistics(self, data_path: str) -> Dict[str, Any]:
        """
        Compute comprehensive statistics from the dataset.
        This is the 'EDA without visualizations' - extracting all numerical insights.
        """
        data_path = Path(data_path)

        # Find all CSV files
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {data_path}")

        logger.info(f"🔍 Found {len(csv_files)} CSV files: {[f.name for f in csv_files]}")

        # Use AI to identify files (NO HARDCODED HEURISTICS!)
        file_identification = self._identify_files_with_ai(data_path, csv_files)

        train_file = file_identification.get("train_file")
        test_file = file_identification.get("test_file")
        submission_file = file_identification.get("submission_file")

        if not train_file:
            raise FileNotFoundError(
                f"AI could not identify training file in {data_path}. "
                f"Files found: {[f.name for f in csv_files]}\n"
                f"AI reasoning: {file_identification.get('reasoning', 'N/A')}"
            )

        logger.info(f"🤖 AI identified files: train={train_file}, test={test_file}, submission={submission_file}")
        if "reasoning" in file_identification:
            logger.info(f"   Reasoning: {file_identification['reasoning']}")

        # Load identified files
        train_path = data_path / train_file
        train = pd.read_csv(train_path)

        test = None
        if test_file:
            test_path = data_path / test_file
            test = pd.read_csv(test_path)

        logger.info(f"📊 Loaded train: {train.shape}, test: {test.shape if test is not None else 'N/A'}")

        # Compute comprehensive statistics
        statistics = {
            # File mapping (NO HARDCODED NAMES!)
            "data_files": {
                "train_file": train_file,
                "test_file": test_file,
                "submission_file": submission_file,
                "all_csv_files": [f.name for f in csv_files]
            },

            # Basic info
            "train_shape": train.shape,
            "test_shape": test.shape if test is not None else None,
            "columns": list(train.columns),

            # Data types
            "dtypes": train.dtypes.astype(str).to_dict(),

            # Missing values
            "missing_values": {
                col: {
                    "count": int(train[col].isnull().sum()),
                    "percentage": float(train[col].isnull().sum() / len(train))
                }
                for col in train.columns
                if train[col].isnull().sum() > 0
            },

            # Numerical statistics
            "numerical_summary": self._get_numerical_summary(train),

            # Categorical statistics
            "categorical_summary": self._get_categorical_summary(train),

            # Correlations (if numerical columns exist)
            "correlations": self._get_correlations(train),

            # Sample data (first 5 rows)
            "sample_data": train.head(5).to_dict(orient='records'),
        }

        return statistics

    def _get_numerical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics for numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numerical_cols:
            return {}

        summary = {}
        for col in numerical_cols:
            try:
                summary[col] = {
                    "mean": float(df[col].mean()),
                    "median": float(df[col].median()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                    "q25": float(df[col].quantile(0.25)),
                    "q75": float(df[col].quantile(0.75)),
                    "skewness": float(df[col].skew()),
                    "kurtosis": float(df[col].kurtosis()),
                }

                # Outlier detection using IQR method
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outlier_threshold_low = q1 - 1.5 * iqr
                outlier_threshold_high = q3 + 1.5 * iqr
                outliers = df[(df[col] < outlier_threshold_low) | (df[col] > outlier_threshold_high)][col]

                summary[col]["outlier_count"] = int(len(outliers))
                summary[col]["outlier_percentage"] = float(len(outliers) / len(df))

            except Exception as e:
                logger.warning(f"⚠️ Could not compute statistics for {col}: {e}")
                summary[col] = {"error": str(e)}

        return summary

    def _get_categorical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistics for categorical columns."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            return {}

        summary = {}
        for col in categorical_cols:
            try:
                value_counts = df[col].value_counts()
                summary[col] = {
                    "unique_count": int(df[col].nunique()),
                    "most_common": value_counts.head(5).to_dict(),
                    "most_common_percentage": float(value_counts.iloc[0] / len(df)) if len(value_counts) > 0 else 0,
                }
            except Exception as e:
                logger.warning(f"⚠️ Could not compute statistics for {col}: {e}")
                summary[col] = {"error": str(e)}

        return summary

    def _get_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute correlation matrix for numerical columns."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) < 2:
            return {}

        try:
            corr_matrix = df[numerical_cols].corr()

            # Convert to dictionary, only including strong correlations (|r| > 0.3)
            strong_correlations = {}
            for i, col1 in enumerate(numerical_cols):
                for j, col2 in enumerate(numerical_cols):
                    if i < j:  # Only upper triangle (avoid duplicates)
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) > 0.3:
                            strong_correlations[f"{col1}_vs_{col2}"] = float(corr_value)

            return strong_correlations

        except Exception as e:
            logger.warning(f"⚠️ Could not compute correlations: {e}")
            return {}

    def _build_analysis_prompt(
        self,
        statistics: Dict[str, Any],
        problem_understanding: Dict[str, Any],
        competition_name: str
    ) -> str:
        """Build prompt for AI analysis."""

        prompt = f"""You are analyzing data for the Kaggle competition: {competition_name}

## Problem Understanding
{json.dumps(problem_understanding, indent=2)}

## Dataset Statistics

### Basic Information
- Train shape: {statistics['train_shape']}
- Test shape: {statistics['test_shape']}
- Columns: {len(statistics['columns'])}

### Column Data Types
{json.dumps(statistics['dtypes'], indent=2)}

### Missing Values
{json.dumps(statistics['missing_values'], indent=2) if statistics['missing_values'] else "No missing values detected"}

### Numerical Columns Summary
{json.dumps(statistics['numerical_summary'], indent=2) if statistics['numerical_summary'] else "No numerical columns"}

### Categorical Columns Summary
{json.dumps(statistics['categorical_summary'], indent=2) if statistics['categorical_summary'] else "No categorical columns"}

### Strong Correlations (|r| > 0.3)
{json.dumps(statistics['correlations'], indent=2) if statistics['correlations'] else "No strong correlations detected"}

### Sample Data (first 5 rows)
{json.dumps(statistics['sample_data'], indent=2)}

---

## Your Task

Analyze the above statistics and provide comprehensive recommendations in **VALID JSON format**.

**IMPORTANT:** Your response must be ONLY valid JSON. No markdown, no code blocks, no explanations outside the JSON.

Return a JSON object with this EXACT structure:

{{
  "data_modality": "tabular|nlp|vision|timeseries|audio|mixed",
  "target_column": "name of target column",
  "target_type": "binary|multiclass|regression|ranking",
  "target_distribution": {{"class_counts or statistics"}},
  "is_imbalanced": true/false,

  "feature_types": {{
    "id_columns": ["list of ID columns to drop"],
    "numerical": ["list of numerical feature columns"],
    "categorical": ["list of categorical feature columns"],
    "text": ["list of text columns if any"],
    "datetime": ["list of datetime columns if any"],
    "drop_candidates": ["columns with too much missing data or not useful"]
  }},

  "data_quality": {{
    "missing_values": {{"column": "percentage or count"}},
    "outliers": ["columns with significant outliers"],
    "high_cardinality_categorical": ["categorical columns with too many unique values"]
  }},

  "preprocessing_required": true/false,

  "preprocessing": {{
    "drop_columns": ["columns to drop"],
    "impute_missing": {{
      "column_name": {{
        "method": "median|mean|mode|constant|drop",
        "reason": "brief explanation"
      }}
    }},
    "encode_categorical": {{
      "column_name": "label|onehot|target",
      "reason": "brief explanation"
    }},
    "scale_numerical": {{
      "method": "standard|minmax|robust|none",
      "columns": ["columns to scale"],
      "reason": "brief explanation"
    }},
    "handle_outliers": {{
      "column_name": {{
        "method": "cap|remove|log_transform|none",
        "reason": "brief explanation"
      }}
    }}
  }},

  "key_insights": [
    "List of important insights from the data analysis",
    "e.g., 'Target is highly imbalanced (80/20 split)'",
    "e.g., 'Age and Fare have strong correlation with target'"
  ]
}}

Remember: Return ONLY the JSON object, nothing else. No markdown code blocks, no explanations before or after.
"""

        return prompt

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from AI response, handling markdown code blocks if present."""
        response = response.strip()

        # Remove markdown code blocks if present
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.rfind("```")
            if start > 6 and end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.rfind("```")
            if start > 2 and end > start:
                response = response[start:end].strip()

        # Parse JSON
        analysis = json.loads(response)

        # Validate required fields
        required_fields = ["data_modality", "preprocessing_required"]
        for field in required_fields:
            if field not in analysis:
                raise ValueError(f"Missing required field in AI response: {field}")

        return analysis

    def _identify_files_with_ai(self, data_path: Path, csv_files: list) -> Dict[str, Any]:
        """
        Use AI to intelligently identify train, test, and submission files.
        NO HARDCODED HEURISTICS - pure AI-based identification.
        """
        logger.info("🤖 Asking AI to identify data files...")

        # Peek at each file to get structure information
        file_previews = {}
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file, nrows=3)  # Just first 3 rows
                file_previews[csv_file.name] = {
                    "columns": list(df.columns),
                    "shape": (len(df), len(df.columns)),
                    "sample_values": df.head(1).to_dict(orient='records')[0] if len(df) > 0 else {}
                }
            except Exception as e:
                logger.warning(f"⚠️ Could not preview {csv_file.name}: {e}")
                file_previews[csv_file.name] = {"error": str(e)}

        # Build prompt for AI
        prompt = f"""Identify the data files for this Kaggle competition.

## Available CSV Files
{json.dumps([f.name for f in csv_files], indent=2)}

## File Previews (first 3 rows)
{json.dumps(file_previews, indent=2)}

## Your Task

Analyze the file names and their column structures to identify:

1. **Training file**: Contains the target variable for training the model
2. **Test file**: Data for making predictions (no target variable, or will be predicted)
3. **Submission format file**: Usually named "sample_submission.csv" or similar, shows required submission format

**Important Rules:**
- Training file MUST have a target/label column
- Test file typically has same features as train but NO target column
- Some competitions have NO test file (you generate predictions from train only)
- Some competitions have multiple train files (X_train.csv + y_train.csv) - pick the main one
- Submission file usually has 2 columns: ID + prediction column

Return ONLY valid JSON with this EXACT structure:

{{
  "train_file": "filename.csv",
  "test_file": "filename.csv or null if no test file",
  "submission_file": "filename.csv or null if not found",
  "auxiliary_files": ["other files if any"],
  "reasoning": "brief explanation of how you identified each file"
}}

**Examples:**

Example 1 (Standard):
Files: ["train.csv", "test.csv", "sample_submission.csv"]
→ {{"train_file": "train.csv", "test_file": "test.csv", "submission_file": "sample_submission.csv", "reasoning": "Clear standard naming"}}

Example 2 (Custom names):
Files: ["application_train.csv", "application_test.csv", "sample_submission.csv"]
→ {{"train_file": "application_train.csv", "test_file": "application_test.csv", "submission_file": "sample_submission.csv", "reasoning": "Train/test indicated by suffixes"}}

Example 3 (No test file):
Files: ["train.csv", "sample_submission.csv"]
→ {{"train_file": "train.csv", "test_file": null, "submission_file": "sample_submission.csv", "reasoning": "No explicit test file, predictions generated from train"}}

Remember: Return ONLY the JSON object. No markdown, no explanations outside JSON.
"""

        # Get AI response
        response = generate_ai_response(self.model, prompt)

        # Parse JSON
        try:
            # Remove markdown if present
            response = response.strip()
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.rfind("```")
                if start > 6 and end > start:
                    response = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.rfind("```")
                if start > 2 and end > start:
                    response = response[start:end].strip()

            file_id = json.loads(response)

            # Validate
            if "train_file" not in file_id:
                raise ValueError("AI response missing 'train_file' field")

            logger.info(f"✅ AI file identification complete")
            return file_id

        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI file identification: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            raise RuntimeError(f"AI did not return valid JSON for file identification: {e}")