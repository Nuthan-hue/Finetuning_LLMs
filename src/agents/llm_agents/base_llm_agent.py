import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class BaseLLMAgent:

    def __init__(
        self,
        name: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ):
        self.name = name
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.provider = os.getenv("AI_PROVIDER", "gemini").lower()

        if self.provider == "gemini":
            self._init_gemini(model_name or "gemini-2.0-flash-exp")
        elif self.provider == "groq":
            self._init_groq(model_name or "llama-3.3-70b-versatile")
        elif self.provider == "openai":
            self._init_openai(model_name or "gpt-4")
        elif self.provider == "kimi":
            self._init_kimi(model_name or "moonshot-v1-8k")
        elif self.provider == "ollama":
            self._init_ollama(model_name or "llama3.3:70b")
        else:
            raise ValueError(f"Unknown AI provider: {self.provider}")

    def _init_gemini(self, model_name: str):
        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")

        genai.configure(api_key=api_key)

        generation_config = {
            "temperature": self.temperature,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
        }

        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=self.system_prompt
        )
        self.model_name = model_name
        logger.info(f"✅ {self.name} with Gemini ({model_name})")

    def _init_groq(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai not installed. Run: pip install openai")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found")

        self.model = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model_name = model_name
        logger.info(f"✅ {self.name} with Groq ({model_name})")

    def _init_openai(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai not installed. Run: pip install openai")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found")

        self.model = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"✅ {self.name} with OpenAI ({model_name})")

    def _init_kimi(self, model_name: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai not installed. Run: pip install openai")

        api_key = os.getenv("KIMI_K2_KEY")
        if not api_key:
            raise ValueError("KIMI_K2_KEY not found")

        self.model = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        self.model_name = model_name
        logger.info(f"✅ {self.name} with Kimi ({model_name})")

    def _init_ollama(self, model_name: str):
        try:
            from ollama import Client
        except ImportError:
            raise RuntimeError("ollama not installed. Run: pip install ollama")

        host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = Client(host=host)
        self.model_name = model_name
        logger.info(f"✅ {self.name} with Ollama ({model_name}) @ {host}")