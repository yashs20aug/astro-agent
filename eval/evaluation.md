# Evaluation: Retrieval Impact Analysis

## Overview

This document demonstrates when RAG retrieval **helps** and when it **hurts**
response quality in the Astro conversational agent. Understanding this distinction
is the core insight behind intent-aware retrieval gating.

---

## Case 1: Retrieval HELPED

### Setup
- **User**: Ritika, Leo (born Aug 20, 1995), first career question
- **Message**: "What should I focus on in my career this month?"
- **Intent classification**: `needs_retrieval = True` (career = knowledge question)

### Retrieved Chunks
| Source | Text | Score |
|--------|------|-------|
| career_guidance | "Opportunities may arise in leadership roles today." | 0.82 |
| zodiac_traits | "Leo: Confident, generous. Strengths: Leadership, charisma, creativity." | 0.78 |
| career_guidance | "Focus on learning to improve long-term gains." | 0.71 |

### Response WITH Retrieval
> As a Leo, your natural leadership abilities are highlighted this month. Mars is
> boosting your drive, and opportunities in leadership roles may present themselves.
> Lean into your charisma in team settings, and focus on skill-building for long-term
> career growth. Avoid rushing major decisions — patience will reward you.

### Response WITHOUT Retrieval (LLM only)
> This month looks positive for your career. Stay focused and work hard on your goals.
> Good things may come your way if you maintain consistency.

### Analysis
The retrieval-augmented response is **specific, personalized, and actionable**:
- References Leo's actual strengths (leadership, charisma) from the knowledge base
- Includes concrete career advice (learning, patience) grounded in the corpus
- Connects zodiac traits to career context naturally

The non-retrieval response is **generic and could apply to anyone**. It's the
astrological equivalent of "you'll have a good day" — technically not wrong,
but not valuable.

**Verdict**: Retrieval transformed a fortune-cookie answer into personalized guidance.

---

## Case 2: Retrieval HURT

### Setup
- **User**: Same session, turn 8 (after discussing career, love, and spirituality)
- **Message**: "Summarize what you've told me so far"
- **Intent classification**: `needs_retrieval = False` (conversational meta-question)

### What Retrieval Would Have Returned (if it ran)
| Source | Text | Score |
|--------|------|-------|
| spiritual_guidance | "Meditation can enhance inner balance and clarity." | 0.45 |
| career_guidance | "Avoid rushing into decisions without clarity." | 0.41 |
| zodiac_traits | "Gemini: Curious, expressive, adaptable." | 0.38 |

### Response WITH Retrieval (bad)
> So far we've discussed your career and love life. Also, meditation can enhance your
> inner balance, and you should avoid rushing into decisions. As a Gemini, you're
> naturally curious... [continues with irrelevant new content]

### Response WITHOUT Retrieval (correct)
> Here's a summary of our conversation: We first discussed your career as a Leo, where
> Mars is boosting your ambition — I advised focusing on leadership and teamwork. Then
> we explored your love life, where Venus plays a key role. Most recently, we discussed
> spiritual growth and the value of meditation. Would you like to dive deeper into any
> of these topics?

### Analysis
Retrieval caused **three distinct problems**:
1. **Injected new advice** into what should have been a pure summary — the user asked
   "what did we discuss" and got unrequested spiritual guidance
2. **Wrong zodiac reference** — retrieved a Gemini chunk when the user is a Leo,
   because "summarize" doesn't carry strong zodiac semantic signal
3. **Wasted tokens and cost** — the embedding API call and 3 retrieved chunks consumed
   ~300 tokens of context window for zero value

**Verdict**: Retrieval added noise, incorrect information, and unnecessary cost.
The intent classifier correctly prevents this by detecting "summarize" as a
conversational pattern.

---

## Trade-off Analysis

### When to Retrieve
| Trigger | Example | Why |
|---------|---------|-----|
| Life area question | "How's my career?" | Knowledge base has specific career guidance |
| Planetary question | "Which planet affects love?" | Needs planetary_impacts data |
| Zodiac personality | "What are Leo's strengths?" | Needs zodiac_traits data |
| New topic area | "Tell me about health" | User exploring new territory |

### When to Skip
| Trigger | Example | Why |
|---------|---------|-----|
| Meta-conversation | "Summarize our chat" | Only needs memory, not knowledge |
| Profile info | "What's my zodiac?" | Already computed, no search needed |
| Language change | "Speak in Hindi" | Preference change, not knowledge query |
| Follow-ups | "Tell me more" | Context already in recent history |
| Greetings | "Hello", "Thanks" | Social turn, no knowledge needed |

### Cost Implications
In a typical 20-turn conversation:
- ~12 turns need retrieval (60%) — these are the knowledge questions
- ~8 turns skip retrieval (40%) — meta, follow-ups, greetings

Per retrieval call:
- Embedding API: ~0.0001 USD (negligible)
- Retrieved chunks: ~200-300 extra prompt tokens
- At GPT-3.5 pricing (~$0.002/1K tokens): ~$0.0005 per retrieval

Over 20 turns, skipping 8 unnecessary retrievals saves:
- ~2,000 prompt tokens
- Marginal cost savings per conversation
- More importantly: **cleaner responses** without irrelevant noise

At scale (10,000 conversations/day), the token savings become meaningful:
- ~20M tokens/day saved
- ~$40/day at GPT-3.5 pricing
- ~$400/day at GPT-4 pricing

### Quality vs. Cost Spectrum
```
Always retrieve ──────────────────────── Never retrieve
  Max grounding        ◄── sweet spot ──►     Zero cost
  Max token cost       (intent-aware)         May hallucinate
  Possible noise                              No noise
```

The intent-aware approach sits in the sweet spot: retrieve when knowledge adds
value, skip when conversation context is sufficient.

---

## Memory Strategy Evaluation

### Sliding Window (implemented as baseline)
- **Pros**: Zero cost, predictable, easy to debug
- **Cons**: Hard cutoff — loses user identity info from turn 1 after ~12 turns

### Summarization (implemented as primary)
- **Pros**: Preserves key facts indefinitely, bounded token growth
- **Cons**: One extra LLM call per compression (~every 12 turns), risk of lossy compression
- **Mitigation**: Explicit instructions in the compression prompt about what to preserve

### Recommendation
Summarization is the right choice for this use case because astrology conversations
are inherently long-running and self-referential. A user who mentioned their career
goals in turn 2 may ask about them again in turn 30. Sliding window alone would
lose that context; summarization preserves it.
