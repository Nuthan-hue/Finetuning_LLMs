"""
Preprocessing Agent
AI-powered agent that generates executable preprocessing code based on data analysis.
"""
import logging
from typing import Dict, Any
from pathlib import Path
from .base_llm_agent import BaseLLMAgent

logger = logging.getLogger(__name__)


class PreprocessingAgent(BaseLLMAgent):
    """AI agent that generates preprocessing code dynamically for any competition."""

    def __init__(self):
        # Load system prompt from file
        prompt_file = Path(__file__).parent.parent / "prompts" / "preprocessing_agent.txt"
        system_prompt = prompt_file.read_text()

        super().__init__(
            name="PreprocessingAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.1,  # Very low temperature for precise code generation
            system_prompt=system_prompt
        )

    async def generate_preprocessing_code(
        self,
        data_analysis: Dict[str, Any],
        data_path: str
    ) -> str:
        """
        Generate executable preprocessing code based on data analysis.

        Args:
            data_analysis: Analysis from DataAnalysisAgent with recommendations
            data_path: Path to raw data directory

        Returns:
            Executable Python code as string
        """
        logger.info("🤖 Generating preprocessing code...")

        # Extract key information
        modality = data_analysis.get("data_modality", "tabular")
        target_column = data_analysis.get("target_column")
        feature_types = data_analysis.get("feature_types", {})
        data_quality = data_analysis.get("data_quality", {})
        preprocessing_rec = data_analysis.get("preprocessing", {})

        context = {
            "modality": modality,
            "target": target_column,
            "data_path": data_path
        }

        prompt = f"""Generate EXECUTABLE Python preprocessing code for this Kaggle competition.

## Competition Details
- **Data Modality**: {modality}
- **Target Column**: {target_column}
- **Data Path**: {data_path}

## Feature Types
{self._format_dict(feature_types)}

## Data Quality Issues
{self._format_dict(data_quality)}

## Preprocessing Recommendations
{self._format_dict(preprocessing_rec)}

## Requirements

Generate a complete Python function that:
1. **Loads data**: Read train.csv and test.csv from data_path
2. **Handles missing values**: Based on recommendations
3. **Encodes categorical variables**: Based on recommendations
4. **Scales numerical features**: If recommended
5. **Processes text/special columns**: If applicable
6. **Drops unnecessary columns**: ID columns, high-missing columns
7. **Saves cleaned data**: clean_train.csv and clean_test.csv
8. **Avoids data leakage**: Fit on train, transform on test
9. **Preserves target column**: Keep target in train, not in test
10. **Returns summary**: Dict with statistics

## Code Template

```python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_data(data_path: str) -> dict:
    \"\"\"
    Preprocess competition data.

    Args:
        data_path: Path to raw data directory

    Returns:
        Dictionary with preprocessing statistics
    \"\"\"
    data_path = Path(data_path)

    # Load data
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "test.csv")

    print(f"Original train shape: {{train.shape}}")
    print(f"Original test shape: {{test.shape}}")

    # Separate target
    target_col = "{target_column}"
    if target_col in train.columns:
        y_train = train[target_col].copy()
        train = train.drop(columns=[target_col])

    # Store IDs if present
    id_columns = {feature_types.get('id_columns', [])}
    test_ids = {{}}
    for id_col in id_columns:
        if id_col in test.columns:
            test_ids[id_col] = test[id_col].copy()

    # Drop ID columns from both
    for id_col in id_columns:
        if id_col in train.columns:
            train = train.drop(columns=[id_col])
        if id_col in test.columns:
            test = test.drop(columns=[id_col])

    # === YOUR PREPROCESSING CODE HERE ===
    # Based on the recommendations above:
    # 1. Handle missing values
    # 2. Encode categorical variables
    # 3. Scale numerical features
    # 4. Process text if needed

    # Example:
    # - For missing numerical: impute with median
    # - For missing categorical: impute with mode or 'missing'
    # - For categorical encoding: use LabelEncoder or one-hot
    # - For scaling: use StandardScaler on numerical columns

    # === END PREPROCESSING ===

    # Add target back to train
    train[target_col] = y_train

    # Add IDs back to test
    for id_col, id_values in test_ids.items():
        test[id_col] = id_values

    # Save cleaned data
    train.to_csv(data_path / "clean_train.csv", index=False)
    test.to_csv(data_path / "clean_test.csv", index=False)

    print(f"Clean train shape: {{train.shape}}")
    print(f"Clean test shape: {{test.shape}}")

    return {{
        "train_shape": train.shape,
        "test_shape": test.shape,
        "columns": list(train.columns),
        "missing_values_remaining": train.isnull().sum().sum()
    }}
```

## Your Task

Fill in the preprocessing section (marked with === YOUR PREPROCESSING CODE HERE ===) with the ACTUAL preprocessing steps based on the recommendations.

**IMPORTANT RULES:**
1. Only use sklearn, pandas, numpy (standard libraries)
2. Fit scalers/encoders on TRAIN only, transform both train and test
3. Handle train and test identically
4. Don't modify target column
5. Don't modify ID columns (we handle them)
6. Write clean, commented code
7. Handle edge cases (missing categories in test, etc.)

Generate the COMPLETE function with the preprocessing section filled in:"""

        # Generate code
        response = await self.reason(prompt, context)

        # Extract code from response
        code = self._extract_code(response)

        logger.info(f"✅ Generated preprocessing code ({len(code)} chars)")

        return code

    def _format_dict(self, d: Dict) -> str:
        """Format dictionary for prompt."""
        if not d:
            return "None specified"

        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"- {key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"- {key}: {', '.join(map(str, value))}")
            else:
                lines.append(f"- {key}: {value}")

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
        if "def preprocess_data" not in code:
            raise RuntimeError(
                "❌ AI did not generate valid preprocessing code. "
                "Expected a 'preprocess_data' function."
            )

        return code