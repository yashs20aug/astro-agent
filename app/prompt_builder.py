"""
Prompt Builder — assembles all context into a structured LLM prompt.

Takes input from:
- User profile (zodiac, name, birth details)
- Conversation memory (summary + recent turns)
- Retrieved knowledge chunks (from RAG)
- Current user message

Outputs OpenAI-compatible chat messages list.
"""


# ---------------------------------------------------------------------------
# System Prompt — defines the bot's persona and rules
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Astro, a warm and knowledgeable Vedic astrology advisor.

Your behavior:
- Give personalized advice based on the user's zodiac sign and birth details.
- Reference planetary influences when relevant to the question.
- Be warm but specific — give actionable guidance, not vague platitudes.
- If retrieved astrological knowledge is provided below, weave it naturally into your response. Do NOT list it mechanically or say "according to my database."
- If no retrieved knowledge is provided, rely on the conversation context and your general astrology understanding.
- Never fabricate specific planetary positions, transits, or dates you don't have data for.

Language rules:
- If the user's preferred language is Hindi, respond entirely in Hindi (Devanagari script).
- If the user's preferred language is English, respond in English.
- If the user mixes languages, respond in the language of their most recent message.

Conversation rules:
- Reference previous topics when relevant to show conversational continuity.
- If asked to summarize, use the conversation summary and recent history — do NOT inject new advice.
- Don't repeat advice already given unless the user asks you to elaborate.
- Keep responses concise (2-4 paragraphs) unless the user asks for detail.
"""


# ---------------------------------------------------------------------------
# Section Builders — each builds one part of the prompt
# ---------------------------------------------------------------------------

def build_profile_block(profile_dict: dict) -> str:
    """Format user profile as a structured block."""
    parts = []
    if profile_dict.get("zodiac"):
        parts.append(f"Zodiac Sign: {profile_dict['zodiac']}")
    if profile_dict.get("name"):
        parts.append(f"Name: {profile_dict['name']}")
    if profile_dict.get("birth_date"):
        parts.append(f"Birth Date: {profile_dict['birth_date']}")
    if profile_dict.get("birth_time"):
        parts.append(f"Birth Time: {profile_dict['birth_time']}")
    if profile_dict.get("birth_place"):
        parts.append(f"Birth Place: {profile_dict['birth_place']}")
    if profile_dict.get("age"):
        parts.append(f"Age: {profile_dict['age']}")

    lang = profile_dict.get("preferred_language", "en")
    lang_name = "Hindi" if lang == "hi" else "English"
    parts.append(f"Preferred Language: {lang_name}")

    return "USER PROFILE:\n" + "\n".join(parts)


def build_summary_block(summary: str) -> str:
    """Format the compressed conversation summary."""
    if not summary:
        return ""
    return f"CONVERSATION HISTORY (SUMMARY):\n{summary}"


def build_retrieval_block(chunks: list[dict]) -> str:
    """Format retrieved knowledge chunks for the prompt."""
    if not chunks:
        return ""

    lines = []
    for chunk in chunks:
        source_label = chunk["source"].replace("_", " ").title()
        lines.append(f"[{source_label}]: {chunk['text']}")

    header = "ASTROLOGICAL KNOWLEDGE (use naturally in your response):"
    return header + "\n" + "\n".join(lines)


def build_history_block(recent_turns: list[dict]) -> str:
    """Format recent conversation turns."""
    if not recent_turns:
        return ""

    lines = []
    for turn in recent_turns:
        role_label = "User" if turn["role"] == "user" else "Astro"
        lines.append(f"{role_label}: {turn['content']}")

    return "RECENT CONVERSATION:\n" + "\n".join(lines)


def build_current_question(message: str) -> str:
    """Format the current user question with a generation prompt."""
    return f"User: {message}\nAstro:"


# ---------------------------------------------------------------------------
# Main Prompt Builder
# ---------------------------------------------------------------------------

class PromptBuilder:
    """
    Assembles all context sections into a single LLM prompt.

    Sections are conditionally included — if no summary exists yet,
    that block is skipped. If retrieval was skipped, that block is omitted.
    This keeps token usage efficient.
    """

    def __init__(self, max_prompt_tokens: int = 3000):
        self.max_prompt_tokens = max_prompt_tokens

    def build(
        self,
        profile_dict: dict,
        memory_context: dict,
        retrieved_chunks: list[dict],
        current_message: str,
    ) -> list[dict]:
        """
        Build the complete prompt as a list of chat messages.

        Args:
            profile_dict: User profile with zodiac, name, etc.
            memory_context: From ConversationMemory.get_context_for_prompt()
            retrieved_chunks: From retrieve_context() — may be empty
            current_message: The user's current question

        Returns:
            List of messages in OpenAI chat format:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        sections = []

        # 1. User profile (always present)
        sections.append(build_profile_block(profile_dict))

        # 2. Conversation summary (if exists)
        summary = memory_context.get("summary", "")
        if summary:
            sections.append(build_summary_block(summary))

        # 3. Retrieved knowledge (if any)
        if retrieved_chunks:
            sections.append(build_retrieval_block(retrieved_chunks))

        # 4. Recent conversation history
        recent = memory_context.get("recent_turns", [])
        if recent:
            sections.append(build_history_block(recent))

        # 5. Current question (always last)
        sections.append(build_current_question(current_message))

        # Combine into user message
        user_content = "\n\n".join(sections)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        return messages

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate for the full prompt."""
        total_chars = sum(len(m["content"]) for m in messages)
        return int(total_chars / 4)  # ~4 chars per token on average
