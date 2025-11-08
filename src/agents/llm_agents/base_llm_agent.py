"""
Base LLM Agent
Provides model initialization for all LLM-powered agents.
"""
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("google-generativeai not installed. Run: pip install google-generativeai")


class BaseLLMAgent:
    """
    Base class for LLM-powered agents.

    Handles only model initialization. All agents directly call
    generate_ai_response() from src.utils.ai_caller for simplicity.
    """

    def __init__(
        self,
        name: str,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLM agent with Google Gemini model.

        Args:
            name: Agent name
            model_name: Gemini model to use
            temperature: Sampling temperature (0.0-1.0)
            system_prompt: System instructions for the agent
        """
        self.name = name
        self.model_name = model_name
        self.temperature = temperature
        self.system_prompt = system_prompt

        # Initialize Gemini
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai not installed")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        genai.configure(api_key=api_key)

        # Configure model
        generation_config = {
            "temperature": temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_prompt
        )

        logger.info(f"✅ Initialized {name} with {model_name}")