"""
Feature Engineering Agent
AI-powered agent that generates executable feature engineering code based on execution plan.
"""
import logging
import json
from typing import Dict, Any, List
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class FeatureEngineeringAgent(BaseLLMAgent):
    """AI agent that generates feature engineering code dynamically for any competition."""

    def __init__(self):
        # Load system prompt from file if exists, otherwise use inline
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "feature_engineering_agent.txt"

        if prompt_file.exists():
            system_prompt = prompt_file.read_text()
        else:
            system_prompt = """You are a Feature Engineering Agent for Kaggle competitions.

Your role:
1. Generate executable Python code for feature engineering
2. Create features based on PlanningAgent recommendations
3. Write clean, efficient, production-ready code
4. Avoid data leakage (fit on train, transform on test)
5. Handle edge cases gracefully

Guidelines:
- Only use standard libraries (pandas, numpy, sklearn, scipy)
- Write well-commented code
- Handle missing values in new features
- Ensure train/test consistency
- Return summary statistics
- NO HARDCODED ASSUMPTIONS - read everything from data_analysis
"""

        super().__init__(
            name="FeatureEngineeringAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.1,  # Very low temperature for precise code generation
            system_prompt=system_prompt
        )

    async def generate_feature_engineering_code(
        self,
        feature_engineering_plan: List[Dict[str, Any]],
        data_analysis: Dict[str, Any],
        clean_data_path: str
    ) -> str:
        """
        Generate executable feature engineering code based on planning recommendations.

        Args:
            feature_engineering_plan: Feature engineering plan from PlanningAgent
            data_analysis: COMPLETE analysis from DataAnalysisAgent (has all info)
            clean_data_path: Path to cleaned data directory

        Returns:
            Executable Python code as string
        """
        logger.info("🤖 Generating feature engineering code...")

        # Format feature plan
        features_description = self._format_feature_plan(feature_engineering_plan)

        # Convert data_analysis to JSON for AI (AI will extract what it needs)
        data_analysis_json = json.dumps(data_analysis, indent=2, default=str)

        prompt = f"""Generate EXECUTABLE Python feature engineering code for this Kaggle competition.

## DATA ANALYSIS (Contains ALL information you need)
```json
{data_analysis_json}
```

## FEATURE ENGINEERING PLAN
{features_description}

## REQUIREMENTS

Create a Python function `engineer_features(data_path: str) -> dict` that:

1. Loads the training and test data
   - Data analysis contains file names, check for cleaned versions
   - Handle missing files gracefully

2. Creates the features listed in the plan above
   - Implement each feature based on the formula/description
   - Apply same transformations to both train and test
   - Handle edge cases (missing values, division by zero, inf, etc.)

3. Preserves data integrity
   - Keep target column in train only (not in test)
   - Keep ID columns in test for submission
   - Ensure train and test have same feature columns

4. Avoids data leakage
   - Fit any transformations on train only
   - Apply fitted transformations to test

5. Saves the featured data
   - Save with clear output names (featured_* pattern recommended)
   - Both train and test files

6. Returns summary statistics
   - train_shape, test_shape
   - feature counts (original, new, added)
   - output file names

## CRITICAL RULES

❌ NO HARDCODED file names - read from data_analysis
❌ NO HARDCODED column names - read from data_analysis
❌ NO ASSUMPTIONS about structure - infer everything
✅ Handle all edge cases gracefully
✅ Add clear comments explaining your logic
✅ Use standard libraries only (pandas, numpy, sklearn, scipy)

Generate the COMPLETE Python code with all imports:"""

        # Generate code
        response = generate_ai_response(self.model, prompt)

        # Extract code from response
        code = self._extract_code(response)

        logger.info(f"✅ Generated feature engineering code ({len(code)} chars)")

        return code

    def _format_feature_plan(self, features: List[Dict[str, Any]]) -> str:
        """Format feature engineering plan for prompt."""
        if not features:
            return "No specific features requested. Generate useful domain-specific features based on data modality."

        lines = []
        for i, feature in enumerate(features, 1):
            name = feature.get("feature_name", f"feature_{i}")
            formula = feature.get("formula", "Not specified")
            reason = feature.get("reason", "Not specified")
            priority = feature.get("priority", "medium")

            lines.append(f"\n**Feature {i}: {name}**")
            lines.append(f"- Formula/Description: {formula}")
            lines.append(f"- Reason: {reason}")
            lines.append(f"- Priority: {priority}")

        return "\n".join(lines)

    def _extract_code(self, response: str) -> str:
        """Extract Python code from AI response."""
        # Remove markdown code blocks if present
        code = response.strip()

        if "```python" in code:
            # Extract code between ```python and ```
            start = code.find("```python") + 9
            end = code.rfind("```")
            if start > 8 and end > start:
                code = code[start:end].strip()
        elif "```" in code:
            # Extract code between ``` and ```
            start = code.find("```") + 3
            end = code.rfind("```")
            if start > 2 and end > start:
                code = code[start:end].strip()

        # Validate it's Python code
        if "def engineer_features" not in code:
            raise RuntimeError(
                "❌ AI did not generate valid feature engineering code. "
                "Expected an 'engineer_features' function."
            )

        return code