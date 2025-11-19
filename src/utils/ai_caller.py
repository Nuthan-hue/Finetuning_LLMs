import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def generate_ai_response(model, prompt: str) -> str:
    provider = os.getenv("AI_PROVIDER", "gemini").lower()

    try:
        if provider == "gemini":
            response = model.generate_content(prompt)
            return response.text.strip()

        elif provider in ["groq", "openai", "kimi"]:
            messages = [{"role": "user", "content": prompt}]
            response = model.chat.completions.create(
                model=_get_model_name(provider, model),
                messages=messages,
                temperature=0.7,
                max_tokens=8192
            )
            return response.choices[0].message.content.strip()

        else:
            raise ValueError(f"Unknown provider: {provider}")

    except Exception as e:
        logger.error(f"AI API call failed: {e}")
        raise


def _get_model_name(provider: str, model) -> str:
    if hasattr(model, 'model_name'):
        return model.model_name

    defaults = {
        "groq": "llama-3.3-70b-versatile",
        "openai": "gpt-4",
        "kimi": "moonshot-v1-8k"
    }
    return defaults.get(provider, "llama-3.3-70b-versatile")


def generate_ai_response_async(model, prompt: str) -> str:
    return generate_ai_response(model, prompt)