# Astro Conversational Insight Agent

A multi-turn conversational AI service for personalized astrology guidance, powered by
Retrieval-Augmented Generation (RAG) with intent-aware retrieval and conversation memory.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run with Stub LLM (No API Key Needed)

```bash
LLM_PROVIDER=stub uvicorn app.main:app --reload
```

### 3. Run with OpenAI

```bash
export OPENAI_API_KEY=sk-your-key-here
export LLM_PROVIDER=openai
uvicorn app.main:app --reload
```

### 4. Test the API

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "abc-123",
    "message": "How will my month be in career?",
    "user_profile": {
      "name": "Ritika",
      "birth_date": "1995-08-20",
      "birth_time": "14:30",
      "birth_place": "Jaipur, India",
      "preferred_language": "hi"
    }
  }'
```

### 5. Run Tests

```bash
pytest tests/ -v
```

### 6. API Docs

Visit `http://localhost:8000/docs` for the interactive Swagger UI.

---

## Architecture

```
POST /chat
    │
    ▼
┌──────────────────┐
│  Profile Layer    │  Compute zodiac from DOB, validate input
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Session/Memory   │  Get or create session, load conversation history
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Intent Classifier│  Rule-based: does this question need retrieval?
└────────┬─────────┘
         │
    ┌────┴────┐
    │ yes     │ no
    ▼         ▼
┌────────┐  (skip)
│ ChromaDB│  Semantic search over astrology knowledge base
│ RAG     │  Filter by similarity threshold
└────┬───┘
     │
     ▼
┌──────────────────┐
│  Prompt Builder   │  Assemble: system + profile + summary + chunks + history + question
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  LLM Provider     │  Stub / OpenAI / Local — swappable via config
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Memory Update    │  Store exchange, trigger compression if needed
└────────┬─────────┘
         │
         ▼
    JSON Response
```

---

## Design Decisions

### Intent-Aware Retrieval

Not every user message benefits from RAG. Retrieving on "summarize our chat" injects
irrelevant astrology chunks into a memory-recall task. The intent classifier uses
pattern matching to gate retrieval:

- **Retrieve**: career/love/spiritual questions, planetary queries, zodiac questions
- **Skip**: summaries, greetings, follow-ups, language changes, profile queries

See `eval/evaluation.md` for detailed case analysis with examples.

### Memory Strategy

Hybrid approach: **sliding window + LLM-based summarization**.

- Recent turns (last 6) are stored verbatim for conversational flow
- Older turns are compressed into a running 2-3 sentence summary
- Compression triggers automatically when history exceeds the window
- Summary preserves: user identity, zodiac, topics discussed, key advice

This ensures bounded token growth while maintaining long-term context. A user who
mentioned career goals in turn 2 can reference them again in turn 30.

### LLM Abstraction

The provider pattern (`LLMProvider` ABC) enables:

- **StubProvider**: Zero-cost testing — full pipeline runs without any API key
- **OpenAIProvider**: Production use with error handling and cost controls
- **LocalModelProvider**: Placeholder for Ollama/llama.cpp (extensibility)

Switch providers via `LLM_PROVIDER` environment variable. The entire system
(retrieval, memory, prompt building) works identically regardless of provider.

### Cost Awareness

- `max_tokens=500` cap on LLM generation (astrology readings don't need essays)
- Intent-aware retrieval skips ~40% of turns, saving ~2,000 tokens per conversation
- Summarization uses `temperature=0.3` for factual compression
- Similarity threshold filters low-relevance chunks to avoid prompt bloat
- Future optimization: use a cheaper model for summarization vs. main chat

### Hindi Toggle

Handled via the system prompt: when `preferred_language` is "hi", the prompt instructs
the LLM to respond entirely in Hindi (Devanagari script). This works reliably with
GPT-3.5/4 without needing a bilingual knowledge base. The stub provider includes
basic Hindi response templates for testing.

---

## Project Structure

```
astro-agent/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, /chat endpoint, wiring
│   ├── profile.py           # Zodiac calculation, user profile management
│   ├── memory.py            # Conversation memory, summarization, sessions
│   ├── retrieval.py         # RAG: ChromaDB, chunking, intent-aware gating
│   ├── llm.py               # LLM abstraction (stub/OpenAI/local)
│   └── prompt_builder.py    # Assembles all context into LLM prompt
├── data/
│   ├── zodiac_traits.json   # All 12 zodiac signs with personality/strengths/challenges
│   ├── planetary_impacts.json # 9 planets (including Rahu/Ketu) with descriptions
│   ├── career_guidance.txt  # Career-specific advice statements
│   ├── love_guidance.txt    # Relationship advice statements
│   ├── spiritual_guidance.txt # Spiritual/meditation guidance
│   └── nakshatra_mapping.json # Bonus: Vedic nakshatra data
├── eval/
│   └── evaluation.md        # Retrieval impact analysis (help vs. hurt cases)
├── tests/
│   └── test_agent.py        # Full test suite (runs without API keys)
├── requirements.txt
└── README.md
```

---

## API Contract

### POST /chat

**Request:**
```json
{
  "session_id": "abc-123",
  "message": "How will my month be in career?",
  "user_profile": {
    "name": "Ritika",
    "birth_date": "1995-08-20",
    "birth_time": "14:30",
    "birth_place": "Jaipur, India",
    "preferred_language": "hi"
  }
}
```

**Response:**
```json
{
  "response": "आपके लिए यह महीना करियर में आगे बढ़ने का है...",
  "zodiac": "Leo",
  "context_used": ["career_guidance", "zodiac_traits"],
  "retrieval_used": true,
  "session_id": "abc-123",
  "turn_number": 2
}
```

### GET /health

Returns service health status and provider info.

---

## Future Enhancements

- **LLM-based intent classification**: Use the LLM itself to classify intent for
  edge cases the rule-based classifier misses
- **Bilingual knowledge base**: Hindi embeddings for native-language retrieval
- **Decay-based memory**: Assign importance scores to turns; profile info decays
  slowly, greetings decay fast
- **Separate summarization model**: Use a cheaper model (GPT-3.5) for memory
  compression while using GPT-4 for main responses
- **Persistent sessions**: Swap in-memory dict for Redis or PostgreSQL
- **Streaming responses**: SSE/WebSocket for real-time token streaming
