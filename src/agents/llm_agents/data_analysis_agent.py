"""
Data Analysis Agent
AI-powered agent that analyzes datasets and suggests preprocessing strategies.
"""
import logging
import json
from typing import Dict, Any, Coroutine
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class DataAnalysisAgent(BaseLLMAgent):
    """AI agent that intelligently analyzes data and suggests preprocessing."""

    def __init__(self):
        # Load system prompt from file
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "data_analysis_agent.txt"
        system_prompt = prompt_file.read_text()

        super().__init__(
            name="DataAnalysisAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.2,  # Very low temperature for precise analysis
            system_prompt=system_prompt
        )

    async def analyze_and_suggest(
        self,
        dataset : str,
        competition_name: str,
        overview_text: str,
    ) -> str:
        """
        Analyze dataset and suggest preprocessing strategies.

        Args:
            dataset_info: Dictionary with dataset statistics
            competition_name: Name of competition

        Returns:
            Dictionary with analysis and recommendations
            :param overview_text:
            :param competition_name:
            :param dataset:
        """
        # Build detailed dataset description


        prompt = f''' Generate a Python module for performing Exploratory Data Analysis (EDA) that can be executed during the runtime of a application. The code should:

1. Be modular and callable via functions or classes
2. Accept a pandas DataFrame as input (not just read from CSV)
3. Log key insights (e.g., missing values, distributions, correlations) using Python’s logging module
4. Avoid blocking visualizations (e.g., use non-interactive backends or save plots to disk)
5. Return structured summaries (e.g., dicts or DataFrames) for downstream agents or components
6. Be lightweight and dependency-safe (use only pandas, numpy, matplotlib/seaborn)
7. Include optional hooks for saving plots or exporting reports
8. Be compatible with agentic orchestration (e.g., callable by other agents)

Assume this module will be used inside a multi-agent AI system solving Kaggle competitions end-to-end. The EDA should be robust, fast, and generalizable across tabular datasets.
'''

        # Get AI analysis
        logger.info(f"🤖 Developing code for EDA for {competition_name}...")
        pythonEADCode = generate_ai_response(self.model, prompt)

        # Extract code from response
        code = self._extract_code(pythonEADCode)

        logger.info(f"✅ Generated EDA code ({len(code)} chars)")

        return code

        # Parse JSON response
        return analysis
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

        return code

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
