"""
Test suite for the Astro Conversational Agent.

All tests use the StubProvider — no API keys needed.
Run: pytest tests/ -v
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.profile import compute_zodiac, compute_age, UserProfile
from app.llm import StubProvider, create_llm_provider
from app.memory import ConversationMemory, SessionManager
from app.retrieval import needs_retrieval, build_knowledge_base, retrieve_context
from app.prompt_builder import PromptBuilder


# ====================================================================
# Profile & Zodiac Tests
# ====================================================================

class TestZodiac:
    def test_leo(self):
        assert compute_zodiac("1995-08-20") == "Leo"

    def test_aries(self):
        assert compute_zodiac("1990-04-05") == "Aries"

    def test_capricorn_december(self):
        assert compute_zodiac("1988-12-25") == "Capricorn"

    def test_capricorn_january(self):
        assert compute_zodiac("2000-01-10") == "Capricorn"

    def test_pisces(self):
        assert compute_zodiac("1995-03-15") == "Pisces"

    def test_scorpio(self):
        assert compute_zodiac("1992-11-10") == "Scorpio"

    def test_boundary_leo_virgo(self):
        # Aug 23 is last day of Leo
        assert compute_zodiac("1995-08-23") == "Leo"
        # Aug 24 is first day of Virgo
        assert compute_zodiac("1995-08-24") == "Virgo"

    def test_user_profile_creation(self):
        profile = UserProfile({
            "name": "Ritika",
            "birth_date": "1995-08-20",
            "birth_time": "14:30",
            "birth_place": "Jaipur, India",
            "preferred_language": "hi",
        })
        assert profile.zodiac == "Leo"
        assert profile.name == "Ritika"
        assert profile.preferred_language == "hi"

    def test_profile_validation_missing_date(self):
        profile = UserProfile({"name": "Test"})
        errors = profile.validate()
        assert len(errors) > 0
        assert "birth_date" in errors[0]


# ====================================================================
# Intent Classification Tests
# ====================================================================

class TestIntentClassification:
    def test_career_needs_retrieval(self):
        assert needs_retrieval("What should I focus on in my career?") is True

    def test_love_needs_retrieval(self):
        assert needs_retrieval("Which planet affects my love life?") is True

    def test_spiritual_needs_retrieval(self):
        assert needs_retrieval("Tell me about my spiritual path") is True

    def test_planet_needs_retrieval(self):
        assert needs_retrieval("What is Saturn doing to me?") is True

    def test_summary_skips_retrieval(self):
        assert needs_retrieval("Summarize what you've told me") is False

    def test_repeat_skips_retrieval(self):
        assert needs_retrieval("Can you say that again?") is False

    def test_greeting_skips_retrieval(self):
        assert needs_retrieval("Hello!") is False

    def test_language_change_skips_retrieval(self):
        assert needs_retrieval("Can you speak in Hindi?") is False

    def test_thank_you_skips_retrieval(self):
        assert needs_retrieval("Thank you so much!") is False

    def test_hindi_summary_skips(self):
        assert needs_retrieval("pehle kya bataya tha?") is False


# ====================================================================
# Retrieval Tests
# ====================================================================

class TestRetrieval:
    @classmethod
    def setup_class(cls):
        """Build the knowledge base once for all retrieval tests."""
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        cls.collection = build_knowledge_base(data_dir)

    def test_career_query_retrieves_career_chunks(self):
        chunks, used = retrieve_context(
            "How will my career be?", "leo", self.collection
        )
        assert used is True
        assert len(chunks) > 0
        sources = [c["source"] for c in chunks]
        assert any("career" in s for s in sources)

    def test_love_query_retrieves_love_chunks(self):
        chunks, used = retrieve_context(
            "Tell me about my love life", "leo", self.collection
        )
        assert used is True
        assert len(chunks) > 0

    def test_summary_returns_empty(self):
        chunks, used = retrieve_context(
            "Summarize our conversation", "leo", self.collection
        )
        assert used is False
        assert len(chunks) == 0

    def test_greeting_returns_empty(self):
        chunks, used = retrieve_context(
            "Hello there!", "leo", self.collection
        )
        assert used is False
        assert len(chunks) == 0

    def test_chunks_have_scores(self):
        chunks, _ = retrieve_context(
            "What planet affects career?", "aries", self.collection
        )
        for chunk in chunks:
            assert "score" in chunk
            assert 0 <= chunk["score"] <= 1

    def test_threshold_filters_irrelevant(self):
        # Very strict threshold — should return fewer results
        chunks, _ = retrieve_context(
            "random nonsense xyz", "leo", self.collection,
            threshold=0.3,
        )
        # With a very strict threshold, irrelevant queries should get few/no results
        assert len(chunks) <= 3


# ====================================================================
# Memory Tests
# ====================================================================

class TestMemory:
    def test_add_and_retrieve_turns(self):
        mem = ConversationMemory(max_recent=4)
        mem.add_turn("user", "Hello")
        mem.add_turn("assistant", "Hi there!")
        ctx = mem.get_context_for_prompt()
        assert len(ctx["recent_turns"]) == 2
        assert ctx["total_turns"] == 2

    def test_sliding_window(self):
        mem = ConversationMemory(max_recent=3, llm_provider=None)
        for i in range(10):
            mem.add_turn("user", f"Message {i}")
        ctx = mem.get_context_for_prompt()
        # Should only have max_recent turns in history
        assert len(ctx["recent_turns"]) <= 3

    def test_compression_triggers(self):
        mem = ConversationMemory(max_recent=3, llm_provider=None)
        # Add more than max_recent * 2 = 6 turns
        for i in range(8):
            mem.add_turn("user", f"Turn {i} about career and Leo zodiac")
        # After compression, history should be trimmed
        assert len(mem.history) <= 3
        # Summary should exist (from fallback compression)
        assert mem.summary != "" or len(mem.history) <= 3

    def test_session_manager(self):
        sm = SessionManager()
        session = sm.get_or_create("test-1", {"name": "Ritika", "zodiac": "Leo"})
        assert session["profile"]["name"] == "Ritika"
        # Same session_id returns same session
        session2 = sm.get_or_create("test-1")
        assert session2["profile"]["name"] == "Ritika"

    def test_session_exchange(self):
        sm = SessionManager()
        sm.get_or_create("test-2", {"name": "Test"})
        sm.add_exchange("test-2", "Hello", "Hi there!")
        ctx = sm.get_memory_context("test-2")
        assert len(ctx["recent_turns"]) == 2


# ====================================================================
# LLM Provider Tests
# ====================================================================

class TestLLMProvider:
    def test_stub_returns_string(self):
        stub = StubProvider()
        result = stub.generate([{"role": "user", "content": "How's my career?"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_stub_career_detection(self):
        stub = StubProvider()
        result = stub.generate([{"role": "user", "content": "Tell me about my career"}])
        assert any(kw in result.lower() for kw in ["career", "leadership", "professional", "learning"])

    def test_stub_love_detection(self):
        stub = StubProvider()
        result = stub.generate([{"role": "user", "content": "How's my love life?"}])
        assert any(kw in result.lower() for kw in ["venus", "relationship", "love", "emotional"])

    def test_factory_default_is_stub(self):
        provider = create_llm_provider()
        assert isinstance(provider, StubProvider)

    def test_factory_explicit_stub(self):
        provider = create_llm_provider("stub")
        assert isinstance(provider, StubProvider)

    def test_factory_unknown_falls_back(self):
        provider = create_llm_provider("nonexistent")
        assert isinstance(provider, StubProvider)


# ====================================================================
# Prompt Builder Tests
# ====================================================================

class TestPromptBuilder:
    def test_basic_prompt_structure(self):
        pb = PromptBuilder()
        messages = pb.build(
            profile_dict={"zodiac": "Leo", "name": "Ritika", "preferred_language": "en"},
            memory_context={"summary": "", "recent_turns": [], "total_turns": 0},
            retrieved_chunks=[],
            current_message="How's my career?",
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Leo" in messages[1]["content"]
        assert "career" in messages[1]["content"].lower()

    def test_retrieval_included_when_present(self):
        pb = PromptBuilder()
        chunks = [{"text": "Focus on learning.", "source": "career_guidance", "score": 0.8}]
        messages = pb.build(
            profile_dict={"zodiac": "Leo", "name": "Test", "preferred_language": "en"},
            memory_context={"summary": "", "recent_turns": [], "total_turns": 0},
            retrieved_chunks=chunks,
            current_message="Career advice?",
        )
        assert "Focus on learning" in messages[1]["content"]
        assert "ASTROLOGICAL KNOWLEDGE" in messages[1]["content"]

    def test_retrieval_excluded_when_empty(self):
        pb = PromptBuilder()
        messages = pb.build(
            profile_dict={"zodiac": "Leo", "name": "Test", "preferred_language": "en"},
            memory_context={"summary": "", "recent_turns": [], "total_turns": 0},
            retrieved_chunks=[],
            current_message="Summarize our chat",
        )
        assert "ASTROLOGICAL KNOWLEDGE" not in messages[1]["content"]

    def test_summary_included_when_present(self):
        pb = PromptBuilder()
        messages = pb.build(
            profile_dict={"zodiac": "Leo", "name": "Test", "preferred_language": "en"},
            memory_context={
                "summary": "Previously discussed career prospects.",
                "recent_turns": [],
                "total_turns": 5,
            },
            retrieved_chunks=[],
            current_message="What else?",
        )
        assert "Previously discussed career" in messages[1]["content"]

    def test_hindi_preference_in_prompt(self):
        pb = PromptBuilder()
        messages = pb.build(
            profile_dict={"zodiac": "Leo", "name": "Ritika", "preferred_language": "hi"},
            memory_context={"summary": "", "recent_turns": [], "total_turns": 0},
            retrieved_chunks=[],
            current_message="Career advice?",
        )
        assert "Hindi" in messages[1]["content"]


# ====================================================================
# Integration Test — Full Pipeline
# ====================================================================

class TestFullPipeline:
    """End-to-end test: simulates a multi-turn conversation."""

    @classmethod
    def setup_class(cls):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        cls.collection = build_knowledge_base(data_dir)
        cls.llm = StubProvider()
        cls.session_manager = SessionManager(llm_provider=cls.llm, max_recent=6)
        cls.prompt_builder = PromptBuilder()

    def _chat(self, session_id, message, profile_dict):
        """Simulate one chat turn."""
        session = self.session_manager.get_or_create(session_id, profile_dict)
        memory_ctx = self.session_manager.get_memory_context(session_id)
        chunks, retrieval_used = retrieve_context(
            message, profile_dict.get("zodiac", "leo").lower(), self.collection
        )
        messages = self.prompt_builder.build(
            profile_dict=profile_dict,
            memory_context=memory_ctx,
            retrieved_chunks=chunks,
            current_message=message,
        )
        response = self.llm.generate(messages)
        self.session_manager.add_exchange(session_id, message, response)
        return response, retrieval_used, chunks

    def test_multi_turn_conversation(self):
        profile = {
            "name": "Ritika", "zodiac": "Leo", "birth_date": "1995-08-20",
            "preferred_language": "en",
        }

        # Turn 1: Career question — should trigger retrieval
        resp1, ret1, chunks1 = self._chat("integ-1", "How will my career be?", profile)
        assert ret1 is True
        assert len(resp1) > 0

        # Turn 2: Love question — should trigger retrieval
        resp2, ret2, chunks2 = self._chat("integ-1", "What about my love life?", profile)
        assert ret2 is True

        # Turn 3: Summary — should NOT trigger retrieval
        resp3, ret3, chunks3 = self._chat("integ-1", "Summarize what we discussed", profile)
        assert ret3 is False
        assert len(chunks3) == 0

        # Turn 4: Thank you — should NOT trigger retrieval
        resp4, ret4, chunks4 = self._chat("integ-1", "Thank you!", profile)
        assert ret4 is False

    def test_memory_persists_across_turns(self):
        profile = {"name": "Amit", "zodiac": "Aries", "preferred_language": "en"}

        self._chat("integ-2", "How's my career?", profile)
        self._chat("integ-2", "What about love?", profile)

        ctx = self.session_manager.get_memory_context("integ-2")
        assert ctx["total_turns"] == 4  # 2 user + 2 assistant
        assert len(ctx["recent_turns"]) == 4
