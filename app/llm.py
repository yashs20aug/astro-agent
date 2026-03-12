"""
LLM Provider Abstraction Layer.

Supports swappable backends:
- StubProvider: Zero-cost canned responses for testing/evaluation
- GroqProvider: Free tier, very fast (Llama/Mixtral on Groq cloud)
- GeminiProvider: Free tier, Google's Gemini model
- OpenAIProvider: Real LLM via OpenAI API (paid)
- LocalModelProvider: Placeholder for local models (Ollama, llama.cpp)

Usage:
    llm = create_llm_provider("stub")    # no API key needed
    llm = create_llm_provider("groq")    # needs GROQ_API_KEY (free)
    llm = create_llm_provider("gemini")  # needs GEMINI_API_KEY (free)
    llm = create_llm_provider("openai")  # needs OPENAI_API_KEY (paid)
"""

import os
import random
from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    @abstractmethod
    def generate(self, messages: list[dict], temperature: float = 0.7) -> str:
        """
        Generate a response from the LLM.

        Args:
            messages: Chat messages in OpenAI format:
                      [{"role": "system/user/assistant", "content": "..."}]
            temperature: Creativity control (0.0 = deterministic, 1.0 = creative)

        Returns:
            Generated text response.
        """
        pass


# ---------------------------------------------------------------------------
# Stub Provider — for testing without any API keys
# ---------------------------------------------------------------------------

class StubProvider(LLMProvider):
    """
    Returns contextually relevant canned responses.
    Scans the prompt for topic keywords to pick an appropriate response.
    Perfect for testing the full pipeline without spending money.
    """

    def __init__(self):
        self._responses = {
            "career": [
                "Your zodiac energy supports professional growth right now. "
                "Focus on leadership opportunities and trust your instincts "
                "when making key decisions this month.",
                "The planetary alignment favors learning and skill-building. "
                "Avoid rushing into major career changes — patience will reward you.",
                "Collaborative projects look especially promising. Your natural "
                "strengths will shine in team settings this period.",
            ],
            "love": [
                "Venus is encouraging emotional openness in your relationships. "
                "Communicate your feelings honestly and appreciate the small gestures.",
                "This is a reflective period for love. Take time to understand "
                "what you truly seek in a relationship before making commitments.",
            ],
            "spiritual": [
                "Meditation and inner reflection are especially powerful for you now. "
                "Focus on gratitude — it attracts abundance into your life.",
                "Your spiritual growth accelerates when you practice mindfulness. "
                "Listen to your intuition; the universe is guiding you.",
            ],
            "health": [
                "Pay attention to rest and recovery this period. Your body needs "
                "balance — combine physical activity with relaxation techniques.",
            ],
            "summary": [
                "Here is a summary of our conversation so far: We have discussed "
                "insights based on your zodiac profile across various life areas. "
                "Let me know if you would like to explore any topic further.",
            ],
            "general": [
                "The stars indicate a period of balance and growth for you. "
                "Stay mindful of your strengths and be patient with challenges. "
                "Would you like guidance on a specific area — career, love, or spiritual?",
            ],
            "hindi_career": [
                "आपकी राशि के अनुसार, यह महीना करियर में आगे बढ़ने का है। "
                "नेतृत्व के अवसरों पर ध्यान दें और अपनी प्रवृत्ति पर भरोसा रखें।",
            ],
            "hindi_love": [
                "शुक्र ग्रह आपके प्रेम जीवन को प्रभावित कर रहा है। "
                "अपनी भावनाओं को खुलकर व्यक्त करें और छोटे इशारों की कद्र करें।",
            ],
            "hindi_general": [
                "सितारे आपके लिए संतुलन और विकास का संकेत दे रहे हैं। "
                "अपनी ताकतों को पहचानें और चुनौतियों में धैर्य रखें।",
            ],
        }

    def generate(self, messages: list[dict], temperature: float = 0.7) -> str:
        full_text = " ".join(m.get("content", "") for m in messages).lower()

        is_hindi = "hindi" in full_text or "preferred language: hindi" in full_text

        # Detect topic
        if any(w in full_text for w in ["summarize", "summary", "so far", "recap"]):
            category = "summary"
        elif any(w in full_text for w in ["career", "job", "work", "professional"]):
            category = "hindi_career" if is_hindi else "career"
        elif any(w in full_text for w in ["love", "relationship", "partner", "venus", "romance"]):
            category = "hindi_love" if is_hindi else "love"
        elif any(w in full_text for w in ["spiritual", "meditation", "soul", "inner", "gratitude"]):
            category = "spiritual"
        elif any(w in full_text for w in ["health", "body", "exercise", "sleep"]):
            category = "health"
        else:
            category = "hindi_general" if is_hindi else "general"

        return random.choice(self._responses[category])


# ---------------------------------------------------------------------------
# OpenAI Provider — real LLM for production use
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    """Real LLM provider using OpenAI's Chat Completions API."""

    def __init__(self, model: str = "gpt-3.5-turbo", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY env var or pass api_key."
            )

    def generate(self, messages: list[dict], temperature: float = 0.7) -> str:
        try:
            import openai

            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=500,  # cost control: astrology readings don't need 2000 words
            )
            return response.choices[0].message.content

        except ImportError:
            return (
                "OpenAI library not installed. Run: pip install openai. "
                "Or use LLM_PROVIDER=stub for testing."
            )
        except Exception as e:
            error_type = type(e).__name__
            return (
                f"I encountered an issue generating a response ({error_type}). "
                "Please try again in a moment."
            )


# ---------------------------------------------------------------------------
# Groq Provider — FREE, very fast (runs Llama/Mixtral on Groq hardware)
# Get your key: https://console.groq.com → sign in with Google → API Keys
# No credit card needed.
# ---------------------------------------------------------------------------

class GroqProvider(LLMProvider):
    """
    Free LLM provider using Groq's API.
    Groq uses the OpenAI-compatible API format, so we reuse the openai library
    with a different base_url. This is a common pattern — many providers do this.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Get one free at https://console.groq.com\n"
                "Then set: GROQ_API_KEY=gsk_your_key_here"
            )

    def generate(self, messages: list[dict], temperature: float = 0.7) -> str:
        try:
            import openai

            # Groq is OpenAI-compatible — same library, different base URL
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.groq.com/openai/v1",
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=500,
            )
            return response.choices[0].message.content

        except ImportError:
            return "OpenAI library not installed. Run: pip install openai"
        except Exception as e:
            error_type = type(e).__name__
            return (
                f"Groq API error ({error_type}): {str(e)[:100]}. "
                "Check your GROQ_API_KEY."
            )


# ---------------------------------------------------------------------------
# Gemini Provider — FREE, Google's model
# Get your key: https://aistudio.google.com/apikey → sign in with Google
# No credit card needed.
# ---------------------------------------------------------------------------

class GeminiProvider(LLMProvider):
    """
    Free LLM provider using Google's Gemini API.
    Uses the google-generativeai library.
    """

    def __init__(self, model: str = "gemini-2.0-flash", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Get one free at https://aistudio.google.com/apikey\n"
                "Then set: GEMINI_API_KEY=your_key_here"
            )

    def generate(self, messages: list[dict], temperature: float = 0.7) -> str:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)

            # Convert OpenAI message format to Gemini format
            # Gemini doesn't have a "system" role — prepend it to the first user message
            system_text = ""
            chat_parts = []

            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg["content"] + "\n\n"
                elif msg["role"] == "user":
                    content = system_text + msg["content"] if system_text else msg["content"]
                    chat_parts.append({"role": "user", "parts": [content]})
                    system_text = ""  # only prepend once
                elif msg["role"] == "assistant":
                    chat_parts.append({"role": "model", "parts": [msg["content"]]})

            response = model.generate_content(
                chat_parts,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=500,
                ),
            )
            return response.text

        except ImportError:
            return (
                "Google Generative AI library not installed. "
                "Run: pip install google-generativeai"
            )
        except Exception as e:
            error_type = type(e).__name__
            return (
                f"Gemini API error ({error_type}): {str(e)[:100]}. "
                "Check your GEMINI_API_KEY."
            )


# ---------------------------------------------------------------------------
# Local Model Provider — placeholder for Ollama / llama.cpp
# ---------------------------------------------------------------------------

class LocalModelProvider(LLMProvider):
    """
    Placeholder for local model inference.
    Demonstrates the abstraction supports any backend.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def generate(self, messages: list[dict], temperature: float = 0.7) -> str:
        raise NotImplementedError(
            "Local model not configured. Use LLM_PROVIDER=stub or LLM_PROVIDER=openai."
        )


# ---------------------------------------------------------------------------
# Factory — create the right provider from config
# ---------------------------------------------------------------------------

def create_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Factory to instantiate the appropriate LLM provider.

    Priority:
        1. Explicit provider_name argument
        2. LLM_PROVIDER environment variable
        3. Falls back to stub (always works, zero config)

    Available providers:
        stub   — canned responses, no API key (for testing)
        groq   — free, fast (needs GROQ_API_KEY)
        gemini — free, Google (needs GEMINI_API_KEY)
        openai — paid (needs OPENAI_API_KEY)
        local  — placeholder for Ollama/llama.cpp
    """
    provider = provider_name or os.getenv("LLM_PROVIDER", "stub")

    if provider == "groq":
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return GroqProvider(model=model)

    elif provider == "gemini":
        model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        return GeminiProvider(model=model)

    elif provider == "openai":
        model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        return OpenAIProvider(model=model)

    elif provider == "local":
        path = os.getenv("LOCAL_MODEL_PATH")
        return LocalModelProvider(model_path=path)

    elif provider == "stub":
        return StubProvider()

    else:
        print(f"[WARNING] Unknown LLM provider '{provider}', falling back to stub.")
        return StubProvider()
