"""
Base LLM Agent
Uses Google Gemini for intelligent decision-making and reasoning.
"""
import os
import logging
import json
from typing import Dict, Any, Optional, List
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
    """Base class for LLM-powered agents using Google Gemini."""

    def __init__(
        self,
        name: str,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize LLM agent.

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
        self.conversation_history: List[Dict[str, str]] = []

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

        logger.info(f"Initialized {name} with {model_name}")

    async def reason(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        structured_output: bool = False
    ) -> str:
        """
        Ask the LLM agent to reason about a problem.

        Args:
            prompt: The question or task for the agent
            context: Additional context as dictionary
            structured_output: Whether to expect JSON output

        Returns:
            Agent's response
        """
        try:
            # Build full prompt with context
            full_prompt = self._build_prompt(prompt, context, structured_output)

            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": full_prompt
            })

            # Generate response
            logger.info(f"{self.name} reasoning about: {prompt[:100]}...")
            response = self.model.generate_content(full_prompt)

            response_text = response.text.strip()

            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })

            logger.info(f"{self.name} response: {response_text[:200]}...")

            return response_text

        except Exception as e:
            logger.error(f"Error in {self.name} reasoning: {e}")
            raise

    async def reason_json(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Ask the agent to reason and return structured JSON output.

        Args:
            prompt: The question or task
            context: Additional context

        Returns:
            Parsed JSON response as dictionary
        """
        response = await self.reason(prompt, context, structured_output=True)

        # Try to extract JSON from response
        try:
            # Remove markdown code blocks if present
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]

            return json.loads(response.strip())

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {self.name}: {e}")
            logger.error(f"Response was: {response}")
            raise ValueError(f"Agent did not return valid JSON: {response[:200]}")

    def _build_prompt(
        self,
        prompt: str,
        context: Optional[Dict[str, Any]],
        structured_output: bool
    ) -> str:
        """Build full prompt with context."""
        parts = []

        # Add context if provided
        if context:
            parts.append("## Context")
            for key, value in context.items():
                parts.append(f"**{key}**: {value}")
            parts.append("")

        # Add main prompt
        parts.append(prompt)

        # Add structured output instruction if needed
        if structured_output:
            parts.append("\n**IMPORTANT**: Respond with valid JSON only. No explanations outside the JSON.")

        return "\n".join(parts)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history