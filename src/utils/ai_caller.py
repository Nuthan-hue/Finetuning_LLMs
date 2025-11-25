import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _get_available_providers():
    """Check which API keys are available in .env"""
    available = []

    if os.getenv("GEMINI_API_KEY"):
        available.append("gemini")
    if os.getenv("GROQ_API_KEY"):
        available.append("groq")
    if os.getenv("OPENAI_API_KEY"):
        available.append("openai")
    if os.getenv("KIMI_API_KEY"):
        available.append("kimi")
    # Ollama is always available if running locally
    if os.getenv("OLLAMA_HOST") or os.getenv("AI_PROVIDER") == "ollama":
        available.append("ollama")

    return available


def _get_primary_provider():
    """Determine primary provider from .env or available keys"""
    # Check if user explicitly set AI_PROVIDER
    env_provider = os.getenv("AI_PROVIDER")
    if env_provider:
        provider = env_provider.lower()

        # Ollama doesn't need API key verification
        if provider == "ollama":
            logger.info(f"🤖 Using OLLAMA (local models)")
            return provider

        # Verify the key exists for cloud providers
        key_map = {
            "gemini": "GEMINI_API_KEY",
            "groq": "GROQ_API_KEY",
            "openai": "OPENAI_API_KEY",
            "kimi": "KIMI_API_KEY"
        }
        if provider in key_map and os.getenv(key_map[provider]):
            logger.info(f"🤖 Using {provider.upper()} (set via AI_PROVIDER)")
            return provider
        else:
            logger.warning(f"⚠️  AI_PROVIDER={provider} but {key_map.get(provider)} not found in .env")

    # Auto-select based on available keys (priority: Ollama > Groq > Gemini > OpenAI > Kimi)
    # Ollama is free and unlimited (local)
    available = _get_available_providers()

    if not available:
        raise ValueError(
            "❌ No AI provider found!\n"
            "Please set at least one of:\n"
            "  - AI_PROVIDER=ollama (recommended, free, local)\n"
            "  - GEMINI_API_KEY\n"
            "  - GROQ_API_KEY (highest free tier limits)\n"
            "  - OPENAI_API_KEY\n"
            "  - KIMI_API_KEY"
        )

    # Priority order: Ollama (local, free, unlimited) > Groq > Gemini > OpenAI > Kimi
    priority = ["ollama", "groq", "gemini", "openai", "kimi"]
    for provider in priority:
        if provider in available:
            logger.info(f"🤖 Auto-selected {provider.upper()} (best available)")
            return provider

    return available[0]


def _ask_ai_for_task_type(prompt: str) -> dict:
    """Use a small, fast AI model to analyze the task and recommend model config

    Args:
        prompt: The user's prompt

    Returns:
        dict with 'task_type', 'reasoning', 'recommended_temperature', 'recommended_model'
    """
    # Use the cheapest, fastest model for meta-analysis
    analysis_prompt = f"""Analyze this AI task and classify it. Return ONLY a JSON object, no other text.

Task prompt:
```
{prompt[:500]}...
```

Classify into ONE of these types:
1. "code" - Generating code, scripts, functions (needs precision, low temperature)
2. "analysis" - Data analysis, examining patterns, identifying insights (medium precision)
3. "planning" - Strategy, recommendations, decision-making (creative, higher temperature)
4. "general" - General conversation, explanations, Q&A

Return JSON format:
{{
    "task_type": "code|analysis|planning|general",
    "reasoning": "brief reason for classification",
    "recommended_temperature": 0.1-0.9,
    "needs_code_model": true|false
}}"""

    try:
        # Priority: Groq (highest rate limits) > Gemini > Others
        # Use Groq if available (14,400 requests/day free tier)
        if os.getenv("GROQ_API_KEY"):
            from groq import Groq
            client = Groq(api_key=os.getenv("GROQ_API_KEY"))
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",  # Fast and accurate
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.1,  # Low temp for precise classification
                max_tokens=200
            )
            response_text = response.choices[0].message.content.strip()

        # Fallback to Gemini if Groq not available
        elif os.getenv("GEMINI_API_KEY"):
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            meta_model = genai.GenerativeModel("gemini-2.0-flash-exp")
            response = meta_model.generate_content(
                analysis_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=200
                )
            )
            response_text = response.text.strip()

        else:
            # No API available for meta-analysis, use fallback
            logger.warning("⚠️  No API key available for task classification, using heuristics")
            return _simple_task_detection(prompt)

        # Parse JSON response
        import json
        import re

        # Extract JSON from markdown code blocks if present
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            response_text = json_match.group(1)
        elif response_text.startswith('```') and response_text.endswith('```'):
            response_text = response_text.strip('`').strip()
            if response_text.startswith('json'):
                response_text = response_text[4:].strip()

        result = json.loads(response_text)
        logger.info(f"🤖 AI Task Classifier: {result['task_type']} - {result['reasoning']}")
        return result

    except Exception as e:
        logger.warning(f"⚠️  AI task classification failed: {e}, using fallback heuristics")

    # Fallback to simple keyword matching
    return _simple_task_detection(prompt)


def _simple_task_detection(prompt: str) -> dict:
    """Fallback simple keyword-based task detection"""
    prompt_lower = prompt.lower()

    code_keywords = ["generate code", "write code", "python", "function", "script", "implement"]
    analysis_keywords = ["analyze", "examine", "identify", "detect", "insights", "patterns"]
    planning_keywords = ["plan", "strategy", "recommend", "suggest", "decide", "choose"]

    if any(kw in prompt_lower for kw in code_keywords):
        return {
            "task_type": "code",
            "reasoning": "Detected code generation keywords",
            "recommended_temperature": 0.1,
            "needs_code_model": True
        }
    elif any(kw in prompt_lower for kw in analysis_keywords):
        return {
            "task_type": "analysis",
            "reasoning": "Detected analysis keywords",
            "recommended_temperature": 0.3,
            "needs_code_model": False
        }
    elif any(kw in prompt_lower for kw in planning_keywords):
        return {
            "task_type": "planning",
            "reasoning": "Detected planning keywords",
            "recommended_temperature": 0.5,
            "needs_code_model": False
        }
    else:
        return {
            "task_type": "general",
            "reasoning": "No specific keywords detected",
            "recommended_temperature": 0.7,
            "needs_code_model": False
        }


def generate_ai_response(model, prompt: str) -> str:
    """Generate AI response using the primary provider

    Uses meta-AI to intelligently select best model for the task.

    Args:
        model: The AI model instance (may be unused if provider is auto-detected)
        prompt: The prompt to send to the AI

    Returns:
        AI response text

    Raises:
        Exception: If the provider fails
    """

    primary_provider = _get_primary_provider()

    # Use AI to analyze the task and recommend model
    task_info = _ask_ai_for_task_type(prompt)

    # Call primary provider
    try:
        return _call_provider(primary_provider, model, prompt, task_info)

    except Exception as e:
        error_str = str(e).lower()

        # Check if it's a rate limit error
        is_rate_limit = any(keyword in error_str for keyword in [
            "rate_limit", "429", "quota", "too many requests", "limit reached"
        ])

        if is_rate_limit:
            logger.error(f"❌ {primary_provider.upper()} rate limit reached")
            logger.error(f"   Please wait for quota to reset or switch providers in .env")
        else:
            logger.error(f"❌ {primary_provider.upper()} error: {e}")

        raise Exception(f"{primary_provider.upper()} failed: {e}")


def _call_provider(provider: str, model, prompt: str, task_info: dict) -> str:
    """Call specific AI provider with task-optimized settings

    Args:
        provider: Provider name (gemini, groq, openai, kimi)
        model: Model instance (may be unused)
        prompt: Prompt text
        task_info: Task information from AI classifier

    Returns:
        AI response text
    """

    # Extract task-specific settings
    temperature = task_info.get("recommended_temperature", 0.7)
    task_type = task_info.get("task_type", "general")

    # Select model based on task and provider
    model_name = _select_model_for_task(provider, task_type)

    logger.info(f"📋 Task: {task_type} | Model: {model_name} | Temp: {temperature}")

    if provider == "gemini":
        # Gemini uses different API
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        gemini_model = genai.GenerativeModel(model_name)
        response = gemini_model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=8192
            )
        )
        return response.text.strip()

    elif provider == "groq":
        # Groq uses OpenAI-compatible API
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=8192
        )
        return response.choices[0].message.content.strip()

    elif provider == "openai":
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=8192
        )
        return response.choices[0].message.content.strip()

    elif provider == "kimi":
        from openai import OpenAI  # Kimi uses OpenAI-compatible API
        client = OpenAI(
            api_key=os.getenv("KIMI_API_KEY"),
            base_url="https://api.moonshot.cn/v1"
        )
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=8192
        )
        return response.choices[0].message.content.strip()

    elif provider == "ollama":
        try:
            from ollama import Client
        except ImportError:
            raise RuntimeError("ollama not installed. Run: pip install ollama")

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        client = Client(host=host)

        response = client.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature}
        )
        return response['message']['content'].strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")


def _select_model_for_task(provider: str, task_type: str) -> str:
    """Select best model for provider and task type

    Args:
        provider: Provider name
        task_type: Task type (code, analysis, planning, general)

    Returns:
        Model name string
    """

    # Model selection matrix: Provider x Task Type
    models = {
        "gemini": {
            "code": "gemini-2.0-flash-exp",  # Fast, good for code
            "analysis": "gemini-2.0-flash-exp",
            "planning": "gemini-2.0-flash-exp",
            "general": "gemini-2.0-flash-exp"
        },
        "groq": {
            "code": "llama-3.3-70b-versatile",  # Best for code
            "analysis": "llama-3.3-70b-versatile",
            "planning": "llama-3.3-70b-versatile",
            "general": "llama-3.3-70b-versatile"
        },
        "openai": {
            "code": "gpt-4-turbo",  # Best for code
            "analysis": "gpt-4",
            "planning": "gpt-4",
            "general": "gpt-3.5-turbo"  # Cheapest for general tasks
        },
        "kimi": {
            "code": "moonshot-v1-32k",  # Larger context
            "analysis": "moonshot-v1-8k",
            "planning": "moonshot-v1-8k",
            "general": "moonshot-v1-8k"
        },
        "ollama": {
            "code": "qwen2.5-coder:14b",  # Specialized code model
            "analysis": "llama3.2:7b",  # Fast for structured output
            "planning": "llama3.3:70b",  # Best reasoning
            "general": "llama3.2:7b"  # Fast and efficient
        }
    }

    return models.get(provider, {}).get(task_type, "default")


def generate_ai_response_async(model, prompt: str) -> str:
    """Async wrapper (currently just calls sync version)"""
    return generate_ai_response(model, prompt)