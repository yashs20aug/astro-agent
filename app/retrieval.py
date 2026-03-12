"""
Retrieval-Augmented Generation (RAG) Layer.

Handles:
- Loading and chunking the knowledge base files
- Embedding and storing in ChromaDB vector store
- Intent-aware retrieval gating (skip retrieval when unnecessary)
- Semantic search with similarity threshold filtering
"""

import json
import os
from typing import Optional

import chromadb


# ---------------------------------------------------------------------------
# Intent Classification — decides if retrieval is needed
# ---------------------------------------------------------------------------

# Patterns that indicate the question is about the conversation itself,
# not about astrology knowledge. Retrieval would be noise here.
SKIP_RETRIEVAL_PATTERNS = [
    "summarize", "summary", "what did you say", "repeat",
    "say that again", "translate", "in hindi", "in english",
    "my name is", "who am i", "what do you know about me",
    "why are you", "what are you", "thank", "hello", "hi ",
    "how are you", "bye", "okay", "ok ", "yes", "no ",
    "can you speak", "change language", "aapka naam",
    "pehle kya bataya", "dobara batao",
]


def needs_retrieval(message: str) -> bool:
    """
    Rule-based intent classifier: should we search the knowledge base?

    Returns False for meta-questions, greetings, profile queries.
    Returns True for astrology/life-area questions that benefit from RAG.

    This is deliberately simple — a more advanced version could use the LLM
    itself to classify intent (see README for discussion).
    """
    message_lower = message.lower().strip()

    # Very short messages are usually conversational
    if len(message_lower.split()) <= 2 and not any(
        kw in message_lower for kw in ["career", "love", "spiritual", "planet"]
    ):
        return False

    for pattern in SKIP_RETRIEVAL_PATTERNS:
        if pattern in message_lower:
            return False

    return True


# ---------------------------------------------------------------------------
# Knowledge Base Loader — chunks files for embedding
# ---------------------------------------------------------------------------

def load_zodiac_traits(filepath: str) -> list[dict]:
    """Load zodiac_traits.json — one chunk per zodiac sign."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sign, traits in data.items():
        text = (
            f"{sign} zodiac sign: {traits['personality']} "
            f"Strengths: {traits['strengths']} "
            f"Challenges: {traits['challenges']}"
        )
        chunks.append({
            "text": text,
            "metadata": {"source": "zodiac_traits", "zodiac": sign.lower(), "topic": "personality"},
            "id": f"zodiac_{sign.lower()}",
        })
    return chunks


def load_planetary_impacts(filepath: str) -> list[dict]:
    """Load planetary_impacts.json — one chunk per planet."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for planet, info in data.items():
        if isinstance(info, dict):
            text = (
                f"{planet}: {info['description']} "
                f"Nature: {info.get('nature', 'unknown')}. "
                f"Influences: {info.get('influences', '')}."
            )
        else:
            text = f"{planet}: {info}"

        chunks.append({
            "text": text,
            "metadata": {"source": "planetary_impacts", "topic": "planet", "planet": planet.lower()},
            "id": f"planet_{planet.lower()}",
        })
    return chunks


def load_text_guidance(filepath: str, source_name: str) -> list[dict]:
    """Load a text guidance file — one chunk per line."""
    chunks = []
    topic = source_name.replace("_guidance", "")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip().lstrip("- ") for line in f if line.strip()]

    for i, line in enumerate(lines):
        chunks.append({
            "text": line,
            "metadata": {"source": source_name, "topic": topic},
            "id": f"{source_name}_{i}",
        })
    return chunks


def load_nakshatra_mapping(filepath: str) -> list[dict]:
    """Load optional nakshatra_mapping.json (bonus)."""
    if not os.path.exists(filepath):
        return []

    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    for nakshatra, info in data.items():
        if isinstance(info, dict):
            text = f"Nakshatra {nakshatra}: {info.get('quality', '')} Ruled by {info.get('ruling_planet', 'unknown')}."
        else:
            text = f"Nakshatra {nakshatra}: {info}"

        chunks.append({
            "text": text,
            "metadata": {"source": "nakshatra_mapping", "topic": "nakshatra"},
            "id": f"nakshatra_{nakshatra.lower()}",
        })
    return chunks


# ---------------------------------------------------------------------------
# Vector Store — ChromaDB setup and population
# ---------------------------------------------------------------------------

def build_knowledge_base(data_dir: str = "data") -> chromadb.Collection:
    """
    Load all knowledge base files, chunk them, and store in ChromaDB.

    Args:
        data_dir: Path to the data/ directory

    Returns:
        A ChromaDB collection ready for querying
    """
    client = chromadb.Client()  # in-memory, no persistence needed

    # Delete collection if it exists (idempotent rebuilds)
    try:
        client.delete_collection("astro_knowledge")
    except Exception:
        pass

    collection = client.create_collection(
        name="astro_knowledge",
        metadata={"hnsw:space": "cosine"},  # use cosine similarity
    )

    # Load all knowledge sources
    all_chunks = []
    all_chunks += load_zodiac_traits(os.path.join(data_dir, "zodiac_traits.json"))
    all_chunks += load_planetary_impacts(os.path.join(data_dir, "planetary_impacts.json"))
    all_chunks += load_text_guidance(os.path.join(data_dir, "career_guidance.txt"), "career_guidance")
    all_chunks += load_text_guidance(os.path.join(data_dir, "love_guidance.txt"), "love_guidance")
    all_chunks += load_text_guidance(os.path.join(data_dir, "spiritual_guidance.txt"), "spiritual_guidance")
    all_chunks += load_nakshatra_mapping(os.path.join(data_dir, "nakshatra_mapping.json"))

    # Add to vector store
    collection.add(
        documents=[c["text"] for c in all_chunks],
        metadatas=[c["metadata"] for c in all_chunks],
        ids=[c["id"] for c in all_chunks],
    )

    print(f"[RAG] Loaded {len(all_chunks)} chunks into vector store.")
    return collection


# ---------------------------------------------------------------------------
# Retrieval Function — the core RAG query
# ---------------------------------------------------------------------------

def retrieve_context(
    message: str,
    user_zodiac: str,
    collection: chromadb.Collection,
    top_k: int = 3,
    threshold: float = 1.5,
) -> tuple[list[dict], bool]:
    """
    Intent-aware retrieval from the knowledge base.

    Steps:
        1. Check if retrieval is needed (intent classification)
        2. Enrich the query with user zodiac for better relevance
        3. Search ChromaDB for semantically similar chunks
        4. Filter results by similarity threshold

    Args:
        message: The user's current question
        user_zodiac: The user's zodiac sign (e.g. "leo")
        collection: ChromaDB collection to search
        top_k: Maximum number of chunks to retrieve
        threshold: Max distance — chunks farther than this are discarded.
                   Lower = stricter. Cosine distance range: [0, 2].

    Returns:
        Tuple of (retrieved_chunks, retrieval_was_used)
        Each chunk: {"text": str, "source": str, "score": float}
    """
    # Step 1: Intent gating
    if not needs_retrieval(message):
        return [], False

    # Step 2: Enrich query with user context
    enriched_query = f"{message} zodiac: {user_zodiac}"

    # Step 3: Semantic search
    results = collection.query(
        query_texts=[enriched_query],
        n_results=top_k,
    )

    # Step 4: Filter by threshold and package results
    chunks = []
    sources_seen = set()

    for doc, distance, metadata in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        if distance < threshold:
            source = metadata.get("source", "unknown")
            chunks.append({
                "text": doc,
                "source": source,
                "score": round(1 - (distance / 2), 3),  # normalize to [0, 1]
            })
            sources_seen.add(source)

    return chunks, len(chunks) > 0


def get_context_sources(chunks: list[dict]) -> list[str]:
    """Extract unique source names from retrieved chunks (for API response)."""
    return list({c["source"] for c in chunks})
