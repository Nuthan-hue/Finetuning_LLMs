"""
Data Validation Agent
AI-powered agent that validates data quality and detects issues.
"""
import logging
import json
import pandas as pd
from typing import Dict, Any
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class DataValidationAgent(BaseLLMAgent):
    """Validates data quality and detects schema issues."""

    def __init__(self):
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "data_validation_agent.txt"
        system_prompt = prompt_file.read_text() if prompt_file.exists() else "You are a data validation expert."

        super().__init__(
            name="DataValidationAgent",
            model_name="llama3.2:3b",  # Lightweight for validation
            temperature=0.1,
            system_prompt=system_prompt
        )

    async def validate_data(
        self,
        data_path: str,
        problem_understanding: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate data quality and detect issues.

        Args:
            data_path: Path to data directory
            problem_understanding: Problem context

        Returns:
            Validation report with issues and fixes
        """
        logger.info("🔍 Validating data quality...")

        data_dir = Path(data_path)
        issues = []
        stats = {}

        # Check files exist
        train_file = data_dir / "train.csv"
        test_file = data_dir / "test.csv"

        if not train_file.exists():
            issues.append({"type": "missing_file", "file": "train.csv", "severity": "critical"})
        if not test_file.exists():
            issues.append({"type": "missing_file", "file": "test.csv", "severity": "critical"})

        if issues:
            return {"valid": False, "issues": issues, "suggested_fixes": ["Download missing files"]}

        # Load and validate
        try:
            train_df = pd.read_csv(train_file, nrows=1000)  # Sample for validation
            test_df = pd.read_csv(test_file, nrows=1000)

            stats = {
                "train_shape": train_df.shape,
                "test_shape": test_df.shape,
                "train_columns": list(train_df.columns),
                "test_columns": list(test_df.columns),
                "common_columns": list(set(train_df.columns) & set(test_df.columns)),
                "train_only": list(set(train_df.columns) - set(test_df.columns)),
                "test_only": list(set(test_df.columns) - set(train_df.columns))
            }

            # Schema mismatch check
            if stats["train_only"] and stats["train_only"] != [problem_understanding.get("target_column", "target")]:
                issues.append({
                    "type": "schema_mismatch",
                    "details": f"Train has extra columns: {stats['train_only']}",
                    "severity": "warning"
                })

        except Exception as e:
            issues.append({"type": "read_error", "error": str(e), "severity": "critical"})

        # AI analysis
        prompt = f"""Analyze this data validation report.

## Problem
{json.dumps(problem_understanding, indent=2)}

## Data Statistics
{json.dumps(stats, indent=2)}

## Detected Issues
{json.dumps(issues, indent=2)}

Return JSON with validation result:
{{
    "valid": true/false,
    "issues": [...],
    "suggested_fixes": ["fix1", "fix2"],
    "can_proceed": true/false
}}"""

        try:
            response = generate_ai_response(self.model, prompt)
            result = json.loads(response.strip().strip("```json").strip("```"))
            logger.info(f"✅ Validation complete: {result.get('can_proceed', False)}")
            return result
        except:
            return {"valid": len(issues) == 0, "issues": issues, "can_proceed": len([i for i in issues if i.get("severity") == "critical"]) == 0}

    def _parse_json(self, text: str) -> Dict:
        cleaned = text.strip().strip("```json").strip("```").strip()
        return json.loads(cleaned)