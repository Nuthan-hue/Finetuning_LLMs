"""
Error Recovery Agent
AI-powered agent that analyzes errors and suggests fixes to recover from failures.
"""
import logging
import json
import traceback
from typing import Dict, Any, Optional
from pathlib import Path
from .base_llm_agent import BaseLLMAgent
from src.utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class ErrorRecoveryAgent(BaseLLMAgent):
    """
    AI agent that analyzes errors and provides recovery strategies.

    Handles code errors, data errors, API failures, and suggests fixes.
    """

    def __init__(self):
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "error_recovery_agent.txt"
        system_prompt = prompt_file.read_text() if prompt_file.exists() else self._get_default_prompt()

        super().__init__(
            name="ErrorRecoveryAgent",
            model_name="qwen2.5-coder:14b",  # Code debugging specialist
            temperature=0.2,  # Low temp for precise debugging
            system_prompt=system_prompt
        )

    def _get_default_prompt(self) -> str:
        return """You are an expert debugging and error recovery specialist.

Your role: Analyze errors, identify root causes, and suggest fixes.

Error types you handle:
1. Python exceptions (syntax, runtime, logic errors)
2. Data errors (missing files, schema mismatches, corrupted data)
3. API errors (rate limits, timeouts, authentication)
4. Model training errors (NaN values, convergence issues)

Provide actionable fixes that can be automatically applied or manually implemented."""

    async def analyze_and_fix(
        self,
        error: Exception,
        context: Dict[str, Any],
        code: Optional[str] = None,
        phase: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Analyze error and suggest recovery strategy.

        Args:
            error: The exception that occurred
            context: Context dict with relevant info
            code: Code that caused the error (if applicable)
            phase: Which phase failed

        Returns:
            Dictionary with:
                - error_type: Classification of error
                - root_cause: Root cause analysis
                - suggested_fix: Recommended fix
                - auto_fixable: Whether can be auto-fixed
                - fixed_code: Fixed code (if applicable)
                - retry_strategy: How to retry
        """
        logger.info(f"🔧 Analyzing error in phase: {phase}")

        # Get full traceback
        tb = traceback.format_exc()

        # Build analysis prompt
        prompt = f"""Analyze this error and suggest a fix.

## Error Information
**Phase**: {phase}
**Error Type**: {type(error).__name__}
**Error Message**: {str(error)}

**Full Traceback**:
```
{tb}
```

## Context
{json.dumps(context, indent=2, default=str)}

{f'''## Code That Failed
```python
{code}
```''' if code else ''}

---

Provide a comprehensive analysis and recovery strategy.

Return ONLY a JSON object:
{{
    "error_type": "syntax|runtime|data|api|model_training",
    "root_cause": "Detailed explanation of what went wrong",
    "suggested_fix": "Step-by-step fix instructions",
    "auto_fixable": true/false,
    "fixed_code": "Corrected code (if auto-fixable and code provided)",
    "retry_strategy": "immediate|with_backoff|skip_phase|manual_intervention",
    "confidence": 0.0-1.0
}}"""

        try:
            response_text = generate_ai_response(self.model, prompt)
            result = self._parse_json_response(response_text)

            logger.info(f"✅ Error analysis complete")
            logger.info(f"   Type: {result['error_type']}")
            logger.info(f"   Auto-fixable: {result.get('auto_fixable', False)}")
            logger.info(f"   Root cause: {result['root_cause'][:100]}...")

            return result

        except Exception as e:
            logger.error(f"❌ Error recovery analysis failed: {e}")
            # Fallback response
            return {
                "error_type": "unknown",
                "root_cause": f"Error recovery agent failed: {str(e)}",
                "suggested_fix": "Manual debugging required",
                "auto_fixable": False,
                "retry_strategy": "manual_intervention",
                "confidence": 0.0
            }

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse AI response into JSON."""
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)

            required = ["error_type", "root_cause", "suggested_fix"]
            for field in required:
                if field not in result:
                    raise ValueError(f"Missing field: {field}")

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.error(f"Response: {response_text[:500]}")
            raise