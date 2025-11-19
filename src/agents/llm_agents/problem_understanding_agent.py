"""
Problem Understanding Agent
AI-powered agent that reads and understands Kaggle competition problems.
"""
import logging
import json
from typing import Dict, Any, Optional
from pathlib import Path

from scripts.seleniumbasedcsrapper import scrape_kaggle_with_selenium
from .base_llm_agent import BaseLLMAgent
from ...utils.ai_caller import generate_ai_response

logger = logging.getLogger(__name__)


class ProblemUnderstandingAgent(BaseLLMAgent):
    """
    AI agent that understands competition problems by reading descriptions.

    Simplified architecture:
    1. Gather competition overview (from cache or web scraping)
    2. Use AI to understand the problem and create strategy
    """

    def __init__(self):
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "problem_understanding_agent.txt"
        system_prompt = prompt_file.read_text()

        super().__init__(
            name="ProblemUnderstandingAgent",
            model_name="gemini-2.0-flash-exp",
            temperature=0.3,
            system_prompt=system_prompt
        )


    async def understand_competition(
        self,
        competition_name: str,
        data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Understand a Kaggle competition by analyzing its description.

        Args:
            competition_name: Name of the Kaggle competition
            data_path: Path to downloaded data (optional, for caching)

        Returns:
            Dictionary containing comprehensive problem understanding:
            {
                "competition_name": str,
                "competition_type": "tabular|nlp|vision|timeseries|audio|multimodal",
                "task_type": "regression|binary_classification|multiclass|...",
                "evaluation_metric": "rmse|accuracy|f1|mAP|...",
                "problem_description": "What we're solving",
                "key_challenges": ["challenge1", "challenge2", ...],
                "recommended_approach": "High-level strategy",
                ...
            }
        """
        logger.info(f"🔍 Understanding competition: {competition_name}")

        # Step 1: Get competition overview text
        overview_text, files_list = self._get_overview(competition_name, data_path)

        # Step 2: Use AI to analyze and understand
        understanding = self._analyze_with_ai(competition_name, self.system_prompt, overview_text, files_list)

        logger.info(f"✅ Problem understood: {understanding['task_type']} using {understanding['evaluation_metric']}")

        return understanding, overview_text

    def _get_overview(
        self,
        competition_name: str,
        data_path: Optional[str] = None
    ) -> tuple[str, str]:
        """
        Get competition overview text from cache or by scraping.

        Args:
            competition_name: Name of the competition
            data_path: Path to data directory (for caching)

        Returns:
            Tuple of (overview_text, files_list)
        """
        logger.info(f"📥 Gathering competition overview...")

        # Get file list from local data directory
        files_list = ""
        if data_path:
            data_dir = Path(data_path)
            if data_dir.exists():
                files = list(data_dir.glob("*"))
                files_list = "\n".join([f.name for f in files if f.is_file()])
                logger.info(f"Found {len(files)} files locally")

        # Check cache for overview
        cache_file = Path(data_path) / f"{competition_name}_overview.txt" if data_path else None

        if cache_file and cache_file.exists():
            logger.info(f"📂 Loading cached overview from {cache_file}")
            overview_text = cache_file.read_text()
        else:
            logger.info(f"🌐 Scraping competition overview from web...")
            # Scrape fresh from Kaggle
            output_dir = str(Path(data_path)) if data_path else None
            overview_text = scrape_kaggle_with_selenium(competition_name, output_dir=output_dir)

            logger.info(f"✅ Scraped overview ({len(overview_text)} chars)")

        return overview_text, files_list

    def _analyze_with_ai(
        self,
        competition_name: str,
        system_prompt: str,
        overview_text: str,
        files_list: str
    ) -> Dict[str, Any]:
        """
        Use AI to analyze competition and create understanding.

        Args:
            competition_name: Name of the competition
            overview_text: Competition description text
            files_list: List of available data files

        Returns:
            Structured understanding dictionary
        """
        # logger.info("🤖 Analyzing competition with AI...")
        prompt_file = Path(__file__).parent.parent.parent / "prompts" / "problem_understanding_prompt.txt"
        system_prompt = prompt_file.read_text()
        system_prompt= f'''You are an expert Kaggle competition analyst. Analyze this competition and provide comprehensive understanding.

# Competition Information

**Competition Name:** {competition_name}

**Competition Overview:**
{overview_text}

**Available Data Files:**
{files_list if files_list else 'Files will be downloaded'}

# Your Task

Analyze this competition and determine:

1. **Competition Type**: What modality? (tabular, nlp, computer_vision, time_series, audio, multimodal)
2. **Task Type**: What are we predicting? (regression, binary_classification, multiclass_classification, object_detection, segmentation, forecasting, etc.)
3. **Evaluation Metric**: What metric is used? (rmse, accuracy, f1, auc, mAP, bleu, etc.)
4. **Problem Description**: What problem are we solving?
5. **Success Criteria**: What makes a good solution?
6. **Key Challenges**: Main challenges to expect
7. **Data Expectations**: What kind of data do we expect?
8. **Submission Requirements**: What format is needed?
9. **Recommended Approach**: High-level strategy

# Context Clues

- Competition name hints at domain (e.g., "titanic" = survival, "house-prices" = regression)
- File names reveal data types (e.g., "train.csv" = tabular, "images.zip" = vision)
- Common patterns:
  - "*-prices*" or "*-sales*" → regression
  - "*-classification*" → classification
  - "*-sentiment*" or "*-nlp*" → NLP
  - "*-image*" or "*-vision*" → computer vision

# Output Format

Respond with ONLY a valid JSON object (no markdown, no code blocks):

{{
    "competition_type": "tabular|nlp|vision|timeseries|audio|multimodal",
    "task_type": "regression|binary_classification|multiclass_classification|...",
    "evaluation_metric": "rmse|accuracy|f1|auc|mAP|bleu|...",
    "metric_description": "Brief explanation of the metric",
    "success_criteria": "What makes a good solution",
    "problem_description": "Clear summary of the problem",
    "key_challenges": ["challenge1", "challenge2", "challenge3"],
    "submission_requirements": {{
        "format": "csv|json|txt|...",
        "expected_columns": ["column1", "column2"],
        "special_requirements": "Any special formatting"
    }},
    "recommended_approach": "High-level strategy",
    "data_expectations": {{
        "expected_data_types": ["tabular|text|images|..."],
        "expected_features": "What features we might see",
        "expected_size": "small|medium|large"
    }},
    "confidence": "high|medium|low"
}}

Provide your analysis:"""
'''

        # Build prompt for AI
        prompt = system_prompt

        try:
            # Get AI analysis
            response_text = generate_ai_response(self.model, prompt)

            # Parse JSON response
            understanding = self._parse_json_response(response_text)

            # Add metadata
            understanding["competition_name"] = competition_name
            understanding["ai_model"] = self.model_name

            logger.info(f"✅ AI analysis complete")
            logger.info(f"   Type: {understanding['competition_type']}")
            logger.info(f"   Task: {understanding['task_type']}")
            logger.info(f"   Metric: {understanding['evaluation_metric']}")

            return understanding

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            raise RuntimeError(
                f"❌ AI failed to understand competition: {str(e)}\n"
                "This is a pure agentic AI system - no fallback logic available."
            )

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse AI response into structured JSON.

        Args:
            response_text: Raw response from AI

        Returns:
            Parsed dictionary
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
            understanding = json.loads(cleaned)

            # Validate required fields
            required_fields = [
                "competition_type",
                "task_type",
                "evaluation_metric",
                "problem_description"
            ]

            for field in required_fields:
                if field not in understanding:
                    raise ValueError(f"Missing required field: {field}")

            return understanding

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response: {response_text[:500]}")
            raise RuntimeError(
                "AI returned invalid JSON. "
                "This is a pure agentic AI system - no fallback parsing available."
            )

    def get_problem_summary(self, understanding: Dict[str, Any]) -> str:
        """
        Generate human-readable summary of problem understanding.

        Args:
            understanding: Problem understanding dictionary

        Returns:
            Formatted summary string
        """
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║  PROBLEM UNDERSTANDING: {understanding['competition_name']:^38} ║
╚══════════════════════════════════════════════════════════════╝

📋 PROBLEM DESCRIPTION
{understanding['problem_description']}

🎯 TASK DETAILS
  Type: {understanding['competition_type']} → {understanding['task_type']}
  Metric: {understanding['evaluation_metric']}
  {understanding.get('metric_description', '')}

✅ SUCCESS CRITERIA
{understanding['success_criteria']}

⚠️  KEY CHALLENGES
"""
        for i, challenge in enumerate(understanding.get('key_challenges', []), 1):
            summary += f"  {i}. {challenge}\n"

        summary += f"""
💡 RECOMMENDED APPROACH
{understanding['recommended_approach']}

📊 DATA EXPECTATIONS
  Types: {', '.join(understanding.get('data_expectations', {}).get('expected_data_types', ['unknown']))}
  Features: {understanding.get('data_expectations', {}).get('expected_features', 'TBD')}
  Size: {understanding.get('data_expectations', {}).get('expected_size', 'unknown')}

📤 SUBMISSION FORMAT
  Format: {understanding.get('submission_requirements', {}).get('format', 'TBD')}
  Columns: {', '.join(understanding.get('submission_requirements', {}).get('expected_columns', ['TBD']))}

🔍 Confidence: {understanding.get('confidence', 'unknown').upper()}
"""
        return summary