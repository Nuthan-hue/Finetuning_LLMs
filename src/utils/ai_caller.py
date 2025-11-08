"""
AI API Caller - Single place for all LLM API calls
This is the ONLY file you need to edit when switching AI providers.

Current: Google Gemini
To switch to another provider (OpenAI, Anthropic, etc.), modify this file only.
"""

import logging

logger = logging.getLogger(__name__)


def generate_ai_response(model, prompt: str) -> str:
    """
    Generate AI response from the model.

    This is the single point where AI API calls happen.
    Edit this function to switch between different AI providers.

    Args:
        model: The AI model instance (currently Google Gemini model)
        prompt: The prompt text to send to the AI

    Returns:
        The generated text response from the AI

    Raises:
        Exception: If the AI call fails

    Example usage in agents:
        from src.utils.ai_caller import generate_ai_response

        response_text = generate_ai_response(self.model, prompt)
    """
    try:
        # ============================================================
        # CURRENT: Google Gemini API call
        # ============================================================
        response = model.generate_content(prompt)
        return response.text.strip()

        # ============================================================
        # TO USE OPENAI: Replace above with:
        # ============================================================
        # messages = [{"role": "user", "content": prompt}]
        # response = model.chat.completions.create(
        #     model="gpt-4",
        #     messages=messages
        # )
        # return response.choices[0].message.content.strip()

        # ============================================================
        # TO USE ANTHROPIC: Replace above with:
        # ============================================================
        # response = model.messages.create(
        #     model="claude-3-5-sonnet-20241022",
        #     messages=[{"role": "user", "content": prompt}],
        #     max_tokens=8192
        # )
        # return response.content[0].text.strip()

    except Exception as e:
        logger.error(f"AI API call failed: {e}")
        raise


def generate_ai_response_async(model, prompt: str) -> str:
    """
    Async version of generate_ai_response.

    Note: Google Gemini SDK doesn't have native async support,
    so this currently wraps the sync call. For true async with other
    providers, you can implement proper async calls here.

    Args:
        model: The AI model instance
        prompt: The prompt text

    Returns:
        The generated text response
    """
    # For now, just call the sync version
    # When switching to providers with async support, implement properly
    return generate_ai_response(model, prompt)