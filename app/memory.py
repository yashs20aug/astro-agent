"""
Conversation Memory Layer.

Implements two strategies:
1. Sliding Window: Keep last N turns (simple, predictable)
2. Summarizing Memory: Compress older turns via LLM summarization

Memory is per-session and controls token growth to stay within LLM context limits.
"""

import time
from typing import Optional

from app.llm import LLMProvider


class ConversationMemory:
    """
    Hybrid memory: sliding window + LLM-based summarization.

    Recent turns are stored verbatim (short-term memory).
    Older turns are compressed into a running summary (long-term memory).
    Compression triggers when history exceeds max_recent * 2.
    """

    def __init__(self, max_recent: int = 6, llm_provider: Optional[LLMProvider] = None):
        """
        Args:
            max_recent: Number of recent turns to keep in full detail.
            llm_provider: LLM used for summarization. If None, falls back to
                          simple truncation (no summarization).
        """
        self.max_recent = max_recent
        self.llm = llm_provider
        self.history: list[dict] = []
        self.summary: str = ""
        self.turn_count: int = 0

    def add_turn(self, role: str, content: str):
        """
        Record a single message (user or assistant).

        Args:
            role: "user" or "assistant"
            content: The message text
        """
        self.history.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
        })
        self.turn_count += 1

        # Trigger compression when history grows too large
        if len(self.history) > self.max_recent * 2:
            self._compress()

    def _compress(self):
        """Summarize older turns and keep only recent ones."""
        # Split: old turns to summarize, recent turns to keep
        old_turns = self.history[: -self.max_recent]
        self.history = self.history[-self.max_recent :]

        if self.llm is None:
            # No LLM available — simple truncation, preserve key facts
            self._compress_without_llm(old_turns)
            return

        # Build the conversation text from old turns
        old_text = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}" for t in old_turns
        )

        summarization_prompt = (
            "You are compressing conversation history for an astrology chatbot.\n"
            "Create a 2-3 sentence summary that MERGES the previous summary with new turns.\n"
            "Do NOT exceed 3 sentences. Replace outdated info with newer info.\n\n"
            "MUST preserve: user name, zodiac sign, birth details, language preference.\n"
            "SHOULD preserve: key advice given, planets discussed, topics covered.\n"
            "CAN drop: greetings, filler, acknowledgments, repeated questions.\n\n"
            f"Previous summary: {self.summary if self.summary else 'None (first compression)'}\n\n"
            f"New turns to incorporate:\n{old_text}\n\n"
            "Updated summary:"
        )

        self.summary = self.llm.generate(
            [{"role": "user", "content": summarization_prompt}],
            temperature=0.3,  # low temperature for factual summarization
        )

    def _compress_without_llm(self, old_turns: list[dict]):
        """Fallback compression: extract key facts without LLM."""
        key_phrases = []
        for turn in old_turns:
            content = turn["content"]
            # Keep messages that look like they contain profile or advice info
            if any(kw in content.lower() for kw in [
                "zodiac", "leo", "aries", "taurus", "gemini", "cancer",
                "virgo", "libra", "scorpio", "sagittarius", "capricorn",
                "aquarius", "pisces", "born", "name is", "career", "love",
                "spiritual", "planet", "mars", "venus", "jupiter", "saturn",
            ]):
                key_phrases.append(f"{turn['role']}: {content[:150]}")

        if key_phrases:
            new_info = " | ".join(key_phrases[-5:])  # keep last 5 key messages
            if self.summary:
                self.summary = f"{self.summary} | {new_info}"
            else:
                self.summary = new_info
            # Prevent summary from growing unbounded
            if len(self.summary) > 500:
                self.summary = self.summary[-500:]

    def get_context_for_prompt(self) -> dict:
        """
        Package memory state for the prompt builder.

        Returns:
            dict with:
                - "summary": compressed history string (may be empty)
                - "recent_turns": list of recent message dicts
                - "total_turns": int count of all turns ever
        """
        return {
            "summary": self.summary,
            "recent_turns": self.history[-self.max_recent :],
            "total_turns": self.turn_count,
        }

    def estimate_tokens(self) -> int:
        """Rough token estimate for budget tracking."""
        summary_tokens = int(len(self.summary.split()) * 1.3) if self.summary else 0
        history_tokens = sum(
            int(len(t["content"].split()) * 1.3) for t in self.history
        )
        return summary_tokens + history_tokens


class SessionManager:
    """
    Manages multiple conversation sessions.

    Each session has its own memory and user profile.
    Sessions are stored in-memory (swap to Redis/DB for production).
    """

    def __init__(self, llm_provider: Optional[LLMProvider] = None, max_recent: int = 6):
        self.sessions: dict[str, dict] = {}
        self.llm = llm_provider
        self.max_recent = max_recent

    def get_or_create(self, session_id: str, user_profile: Optional[dict] = None) -> dict:
        """Get an existing session or create a new one."""
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "memory": ConversationMemory(
                    max_recent=self.max_recent,
                    llm_provider=self.llm,
                ),
                "profile": user_profile or {},
                "created_at": time.time(),
            }
        elif user_profile:
            # Merge new profile data (user might add details over time)
            self.sessions[session_id]["profile"].update(user_profile)

        return self.sessions[session_id]

    def add_exchange(self, session_id: str, user_msg: str, assistant_msg: str):
        """Record a full user→assistant exchange."""
        session = self.sessions[session_id]
        session["memory"].add_turn("user", user_msg)
        session["memory"].add_turn("assistant", assistant_msg)

    def get_memory_context(self, session_id: str) -> dict:
        """Get the memory context for prompt building."""
        session = self.sessions[session_id]
        return session["memory"].get_context_for_prompt()
