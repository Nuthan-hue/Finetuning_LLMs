"""
Feature Engineering Agent - Generates and executes feature engineering code

This agent reads a feature engineering plan and generates Python code to create new features.
It's modality-aware and can handle tabular, NLP, and other data types.
"""
import logging
import json
from typing import Dict, Any, List
from pathlib import Path
import subprocess
import sys

from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseLLMAgent):
    """
    AI agent that generates executable Python code for feature engineering.

    This agent is truly AI-driven:
    - Reads feature plan from PlanningAgent
    - Generates Python code to create those features
    - Executes the code on clean data
    - Returns paths to featured datasets
    """

    def __init__(self):
        # Load system prompt
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "feature_engineering_agent.txt"
        system_prompt = prompt_file.read_text() if prompt_file.exists() else self._get_default_system_prompt()

        super().__init__(
            name="FeatureEngineeringAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.1,  # Low temperature for precise code generation
            system_prompt=system_prompt
        )

    def _get_default_system_prompt(self) -> str:
        return """You are an expert feature engineering agent.

Your role: Generate executable Python code to create features from data.

You receive a feature engineering plan and generate clean, executable Python code that:
1. Loads the data
2. Creates the specified features
3. Handles edge cases (missing values, divisions by zero)
4. Saves the featured datasets

Be precise, handle errors gracefully, and create robust code."""

    async def generate_features(
        self,
        feature_plan: List[Dict[str, Any]],
        clean_data_path: str,
        data_modality: str,
        target_column: str = None
    ) -> Dict[str, Any]:
        """
        Generate and execute feature engineering code.

        Args:
            feature_plan: List of features to create from PlanningAgent
                Example: [
                    {
                        "feature_name": "family_size",
                        "formula": "SibSp + Parch + 1",
                        "reason": "Capture family unit effect",
                        "priority": 1
                    }
                ]
            clean_data_path: Path to clean data directory
            data_modality: tabular/nlp/vision/etc
            target_column: Name of target column (to exclude from features)

        Returns:
            {
                "feature_engineering_code": str,
                "featured_train_path": str,
                "featured_test_path": str,
                "features_created": List[str],
                "execution_log": str
            }
        """
        logger.info(f"🔧 Generating feature engineering code...")
        logger.info(f"   Data modality: {data_modality}")
        logger.info(f"   Features to create: {len(feature_plan)}")

        # Build prompt for AI
        prompt = self._build_code_generation_prompt(
            feature_plan=feature_plan,
            clean_data_path=clean_data_path,
            data_modality=data_modality,
            target_column=target_column
        )

        # Get AI-generated code
        response = generate_ai_response(self.model, prompt)

        # Parse code from response
        code = self._extract_code(response)

        logger.info(f"✅ Feature engineering code generated ({len(code)} chars)")

        # Save code to file
        code_file = Path(clean_data_path) / "feature_engineering.py"
        code_file.write_text(code)
        logger.info(f"💾 Code saved to: {code_file}")

        # Execute the code
        logger.info("⚙️  Executing feature engineering code...")
        execution_log = self._execute_code(code_file, clean_data_path)

        # Determine output paths
        clean_path = Path(clean_data_path)
        featured_train = clean_path / "featured_train.csv"
        featured_test = clean_path / "featured_test.csv"

        # Extract features created
        features_created = [f["feature_name"] for f in feature_plan]

        result = {
            "feature_engineering_code": code,
            "featured_train_path": str(featured_train),
            "featured_test_path": str(featured_test),
            "features_created": features_created,
            "execution_log": execution_log,
            "code_file": str(code_file)
        }

        logger.info(f"✅ Feature engineering completed")
        logger.info(f"   Created {len(features_created)} features")
        logger.info(f"   Output: {featured_train}")

        return result

    def _build_code_generation_prompt(
        self,
        feature_plan: List[Dict[str, Any]],
        clean_data_path: str,
        data_modality: str,
        target_column: str = None
    ) -> str:
        """Build prompt for AI to generate feature engineering code."""

        # Format feature plan
        features_description = "\n".join([
            f"{i+1}. {f['feature_name']}: {f.get('formula', f.get('description', 'N/A'))}"
            f"   Reason: {f.get('reason', 'N/A')}"
            for i, f in enumerate(feature_plan)
        ])

        prompt = f"""Generate executable Python code for feature engineering.

## TASK
Create new features from the data according to the plan below.

## DATA INFORMATION
- Data path: {clean_data_path}
- Data modality: {data_modality}
- Target column: {target_column or "Unknown"}
- Input files: clean_train.csv, clean_test.csv (in the data path)
- Output files: featured_train.csv, featured_test.csv (save to same path)

## FEATURE ENGINEERING PLAN
{features_description}

## REQUIREMENTS

1. **Load Data:**
   ```python
   import pandas as pd
   import numpy as np
   from pathlib import Path

   data_path = Path("{clean_data_path}")
   train = pd.read_csv(data_path / "clean_train.csv")
   test = pd.read_csv(data_path / "clean_test.csv")
   ```

2. **Create Features:**
   - Implement each feature from the plan
   - Handle missing values gracefully (fill or skip)
   - Handle division by zero (add small epsilon or use np.where)
   - Apply same transformations to both train and test
   - Keep all original columns

3. **Save Featured Data:**
   ```python
   train.to_csv(data_path / "featured_train.csv", index=False)
   test.to_csv(data_path / "featured_test.csv", index=False)
   print(f"✅ Created {{len(train.columns)}} total columns")
   print(f"✅ Train shape: {{train.shape}}")
   print(f"✅ Test shape: {{test.shape}}")
   ```

4. **Error Handling:**
   - Wrap in try-except blocks
   - Print clear error messages
   - Don't fail silently

5. **Code Quality:**
   - Use clear variable names
   - Add comments for complex operations
   - Handle edge cases
   - Be defensive (check for column existence)

## MODALITY-SPECIFIC GUIDELINES

{"### Tabular Data:" if data_modality == "tabular" else ""}
{"- Interaction terms: col1 * col2" if data_modality == "tabular" else ""}
{"- Polynomial features: col ** 2" if data_modality == "tabular" else ""}
{"- Binning: pd.cut() or pd.qcut()" if data_modality == "tabular" else ""}
{"- Aggregations: groupby().mean()" if data_modality == "tabular" else ""}

{"### NLP Data:" if data_modality == "nlp" else ""}
{"- Text length features: text.str.len()" if data_modality == "nlp" else ""}
{"- Word count: text.str.split().str.len()" if data_modality == "nlp" else ""}
{"- Character counts: text.str.count('pattern')" if data_modality == "nlp" else ""}

## OUTPUT FORMAT

Return ONLY the Python code, no explanations.

```python
# Your complete, executable code here
```

Remember:
- The code must be completely standalone and executable
- All imports must be included
- Must work with the exact file paths provided
- Must save both featured_train.csv and featured_test.csv
"""

        return prompt

    def _extract_code(self, response: str) -> str:
        """Extract Python code from AI response."""
        response = response.strip()

        # Remove markdown code blocks
        if "```python" in response:
            start = response.find("```python") + 9
            end = response.rfind("```")
            if start > 8 and end > start:
                code = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.rfind("```")
            if start > 2 and end > start:
                code = response[start:end].strip()
        else:
            # No markdown, assume entire response is code
            code = response

        return code

    def _execute_code(self, code_file: Path, clean_data_path: str) -> str:
        """
        Execute the generated Python code.

        Args:
            code_file: Path to the Python file to execute
            clean_data_path: Path to data directory (for context)

        Returns:
            Execution log (stdout + stderr)
        """
        try:
            # Execute the code
            result = subprocess.run(
                [sys.executable, str(code_file)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                cwd=str(Path(clean_data_path).parent)  # Run from parent dir
            )

            log = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n\nReturn code: {result.returncode}"

            if result.returncode != 0:
                logger.warning(f"⚠️  Code execution had non-zero return code: {result.returncode}")
                logger.warning(f"STDERR: {result.stderr}")
            else:
                logger.info("✅ Code executed successfully")

            return log

        except subprocess.TimeoutExpired:
            error_msg = "❌ Code execution timeout (5 minutes)"
            logger.error(error_msg)
            return error_msg

        except Exception as e:
            error_msg = f"❌ Code execution error: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main run method for agent interface.

        Args:
            context: Dictionary containing:
                - feature_plan: List of features to create
                - clean_data_path: Path to clean data
                - data_modality: tabular/nlp/etc
                - target_column: Name of target (optional)

        Returns:
            Dictionary with featured data paths and metadata
        """
        return await self.generate_features(
            feature_plan=context.get("feature_plan", []),
            clean_data_path=context["clean_data_path"],
            data_modality=context.get("data_modality", "tabular"),
            target_column=context.get("target_column")
        )