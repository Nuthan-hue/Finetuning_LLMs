"""
Problem Understanding Agent
AI-powered agent that reads and understands Kaggle competition problems BEFORE analyzing data.
"""
import logging
import json
from scripts.seleniumbasedcsrapper import scrape_kaggle_with_selenium
import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
from pathlib import Path
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)


class ProblemUnderstandingAgent:
    """
    AI agent that understands competition problems by reading descriptions and requirements.

    This agent operates BEFORE data analysis to understand:
    - What problem we're solving
    - What success looks like
    - What constraints exist
    - What the evaluation criteria are
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize the Problem Understanding Agent.

        Args:
            model_name: Gemini model to use for understanding
        """
        self.model_name = model_name
        self._setup_llm()

    def _setup_llm(self):
        """Set up Gemini LLM."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "❌ GEMINI_API_KEY not found. "
                "This is a pure agentic AI system - AI analysis is required. "
                "Set GEMINI_API_KEY in your environment."
            )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)
        logger.info(f"✅ Initialized Gemini model: {self.model_name}")

    async def understand_competition(
        self,
        competition_name: str,
        data_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Understand a Kaggle competition by reading its description and requirements.

        Args:
            competition_name: Name of the Kaggle competition
            data_path: Path to downloaded data (from Phase 1)

        Returns:
            Dictionary containing comprehensive problem understanding:
            {
                "competition_name": str,
                "competition_type": "tabular|nlp|vision|timeseries|audio|multimodal",
                "task_type": "regression|binary_classification|multiclass|...",
                "evaluation_metric": "rmse|accuracy|f1|mAP|...",
                "metric_description": "What the metric measures",
                "success_criteria": "What makes a good solution",
                "problem_description": "Summary of what we're trying to solve",
                "key_challenges": ["challenge1", "challenge2", ...],
                "submission_requirements": {
                    "format": "csv|json|...",
                    "columns": ["col1", "col2", ...],
                    "special_requirements": "..."
                },
                "timeline": {
                    "start_date": "...",
                    "end_date": "...",
                    "days_remaining": int
                },
                "recommended_approach": "High-level strategy",
                "data_expectations": {
                    "expected_data_types": ["tabular", "text", "images", ...],
                    "expected_features": "What kind of features to expect",
                    "expected_size": "Small/medium/large dataset"
                }
            }
        """
        logger.info(f"🔍 Understanding competition: {competition_name}")

        # Step 1: Fetch competition information from local data and Kaggle
        competition_info = await self._fetch_competition_info(competition_name, data_path)

        if not competition_info:
            raise RuntimeError(
                f"❌ Failed to fetch competition info for: {competition_name}"
            )

        # Step 2: Use AI to understand the problem
        understanding = await self._ai_understand_problem(
            competition_name,
            competition_info
        )

        logger.info(f"✅ Problem understood: {understanding['task_type']} using {understanding['evaluation_metric']}")

        return understanding

    async def _fetch_competition_info(
        self,
        competition_name: str,
        data_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch competition information from local data and Kaggle web.

        Args:
            competition_name: Name of the competition
            data_path: Path to downloaded data (from Phase 1)

        Returns:
            Dictionary with competition details or None if failed
        """
        try:
            logger.info(f"📥 Gathering competition information...")

            # Get file list from local data directory (already downloaded in Phase 1)
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
                logger.info(f"📂 Loading cached overview")
                overview_text = cache_file.read_text()
                print(repr(overview_text))
            else:
                # Scrape overview page for real competition description
                # Pass data_path directory so selenium saves to the correct location
                output_dir = str(Path(data_path)) if data_path else None
                overview_text = scrape_kaggle_with_selenium(competition_name, output_dir=output_dir)

                # Cache it (if not already saved by selenium)
                if cache_file and overview_text and not cache_file.exists():
                    cache_file.write_text(overview_text)
                    logger.info(f"💾 Cached overview to {cache_file.name}")

            competition_info = {
                "name": competition_name,
                "files_list": files_list,
                "overview": overview_text
            }

            logger.info(f"✅ Gathered competition info")
            return competition_info

        except Exception as e:
            logger.error(f"Error gathering competition info: {e}")
            return None

    async def _ai_understand_problem(
        self,
        competition_name: str,
        competition_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Gemini AI to understand the competition problem.

        Args:
            competition_name: Name of the competition
            competition_info: Raw competition information from Kaggle

        Returns:
            Structured understanding of the problem
        """
        logger.info("🤖 Using AI to understand the problem...")

        # Build comprehensive prompt for AI
        prompt = self._build_understanding_prompt(competition_name, competition_info)

        try:
            # Get AI analysis
            response = self.model.generate_content(prompt)

            # Parse JSON response
            understanding = self._parse_ai_response(response.text)

            # Add metadata
            understanding["competition_name"] = competition_name
            understanding["ai_model"] = self.model_name

            return understanding

        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            raise RuntimeError(
                f"❌ AI failed to understand competition problem: {str(e)}\n"
                "This is a pure agentic AI system - no fallback logic available."
            )

    def _build_understanding_prompt(
        self,
        competition_name: str,
        competition_info: Dict[str, Any]
    ) -> str:
        """
        Build comprehensive prompt for AI to understand the problem.

        Args:
            competition_name: Name of competition
            competition_info: Competition information

        Returns:
            Formatted prompt string
        """
        overview = competition_info.get('overview', '')
        overview_section = f"\n**Competition Overview:**\n{overview}\n" if overview else ""

        return f"""You are an expert Kaggle competition analyst. Your task is to understand a competition problem BEFORE analyzing any data.

# Competition Information

**Competition Name:** {competition_name}
{overview_section}
**Data Files:**
{competition_info.get('files_list', 'Competition files will be analyzed')}

# Your Task

Analyze this competition and provide a comprehensive understanding of the problem. Based on the competition name, file names, and any available information, infer:

1. **Competition Type**: What modality is this? (tabular, nlp, computer_vision, time_series, audio, multimodal)
2. **Task Type**: What are we predicting? (regression, binary_classification, multiclass_classification, object_detection, segmentation, forecasting, etc.)
3. **Evaluation Metric**: What metric is likely used? (rmse, accuracy, f1, auc, mAP, bleu, etc.)
4. **Problem Description**: What problem are we trying to solve?
5. **Success Criteria**: What makes a good solution?
6. **Key Challenges**: What are the main challenges?
7. **Data Expectations**: What kind of data do we expect?
8. **Submission Requirements**: What format is needed?
9. **Recommended Approach**: High-level strategy

# Important Context Clues

- Competition name often hints at the domain (e.g., "titanic" = survival prediction, "house-prices" = regression)
- File names reveal data types (e.g., "train.csv" = tabular, "images.zip" = vision)
- Common patterns:
  - "*-prices*" or "*-sales*" → regression
  - "*-classification*" or "*-detection*" → classification
  - "*-sentiment*" or "*-nlp*" → NLP
  - "*-image*" or "*-vision*" → computer vision
  - "*-forecast*" or "*-timeseries*" → time series

# Output Format

Respond with ONLY a valid JSON object (no markdown, no code blocks):

{{
    "competition_type": "tabular|nlp|vision|timeseries|audio|multimodal",
    "task_type": "regression|binary_classification|multiclass_classification|object_detection|...",
    "evaluation_metric": "rmse|accuracy|f1|auc|mAP|bleu|...",
    "metric_description": "Brief explanation of what this metric measures",
    "success_criteria": "What makes a good solution for this competition",
    "problem_description": "Clear summary of what we're trying to solve",
    "key_challenges": ["challenge1", "challenge2", "challenge3"],
    "submission_requirements": {{
        "format": "csv|json|txt|...",
        "expected_columns": ["predicted_column_name"],
        "special_requirements": "Any special formatting needs"
    }},
    "recommended_approach": "High-level strategy for tackling this problem",
    "data_expectations": {{
        "expected_data_types": ["tabular|text|images|audio|..."],
        "expected_features": "What kind of features we might see",
        "expected_size": "small|medium|large"
    }},
    "confidence": "high|medium|low"
}}

Provide your analysis:"""

    def _parse_ai_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse AI response into structured format.

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

            logger.info(f"✅ AI understanding parsed successfully")
            logger.info(f"   Type: {understanding['competition_type']}")
            logger.info(f"   Task: {understanding['task_type']}")
            logger.info(f"   Metric: {understanding['evaluation_metric']}")

            return understanding

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse AI response as JSON: {e}")
            logger.error(f"Response: {response_text[:500]}")
            raise RuntimeError(
                "AI returned invalid JSON response. "
                "This is a pure agentic AI system - no fallback parsing available."
            )
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            raise

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
