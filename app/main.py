"""
Astro Conversational Insight Agent — FastAPI Application.

Wires together all layers:
    Profile → Memory → Retrieval → Prompt Builder → LLM → Response

Run:
    uvicorn app.main:app --reload

Test with stub (no API key):
    LLM_PROVIDER=stub uvicorn app.main:app --reload

Test with OpenAI:
    OPENAI_API_KEY=sk-... LLM_PROVIDER=openai uvicorn app.main:app --reload
"""

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from app.profile import UserProfile
from app.llm import create_llm_provider
from app.memory import SessionManager
from app.retrieval import build_knowledge_base, retrieve_context, get_context_sources
from app.prompt_builder import PromptBuilder


# ---------------------------------------------------------------------------
# Request / Response Models (matches the API contract in the assignment)
# ---------------------------------------------------------------------------

class UserProfileInput(BaseModel):
    name: str = "User"
    birth_date: str = Field(..., description="Birth date in YYYY-MM-DD format")
    birth_time: Optional[str] = Field(None, description="Birth time in HH:MM format")
    birth_place: Optional[str] = Field(None, description="Birth city/location")
    preferred_language: str = Field("en", description="'en' or 'hi'")


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(..., description="User's message/question")
    user_profile: UserProfileInput


class ChatResponse(BaseModel):
    response: str
    zodiac: str
    context_used: list[str]
    retrieval_used: bool
    session_id: str
    turn_number: int


# ---------------------------------------------------------------------------
# Application State (initialized on startup)
# ---------------------------------------------------------------------------

# These are module-level so they persist across requests
llm_provider = None
session_manager = None
knowledge_collection = None
prompt_builder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup."""
    global llm_provider, session_manager, knowledge_collection, prompt_builder

    print("[STARTUP] Initializing Astro Agent...")

    # 1. LLM Provider
    provider_name = os.getenv("LLM_PROVIDER", "stub")
    llm_provider = create_llm_provider(provider_name)
    print(f"[STARTUP] LLM Provider: {type(llm_provider).__name__}")

    # 2. Session Manager (with LLM for summarization)
    session_manager = SessionManager(llm_provider=llm_provider, max_recent=6)
    print("[STARTUP] Session Manager ready")

    # 3. Knowledge Base (ChromaDB)
    data_dir = os.getenv("DATA_DIR", "data")
    knowledge_collection = build_knowledge_base(data_dir)

    # 4. Prompt Builder
    prompt_builder = PromptBuilder(max_prompt_tokens=3000)
    print("[STARTUP] Prompt Builder ready")

    print("[STARTUP] Astro Agent initialized successfully!\n")
    yield  # app is running
    print("[SHUTDOWN] Astro Agent shutting down.")


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Astro Conversational Insight Agent",
    description="RAG-powered multi-turn astrology chatbot with conversation ownership",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the chat UI (served from file:// or any localhost port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.

    Flow:
        1. Validate input & compute zodiac from birth date
        2. Get or create session (with conversation memory)
        3. Decide if retrieval is needed (intent-aware gating)
        4. If yes, retrieve relevant knowledge chunks
        5. Build the full prompt (profile + memory + chunks + question)
        6. Generate response via LLM
        7. Store the exchange in memory
        8. Return response with metadata
    """
    start_time = time.time()

    # --- Step 1: Profile & Validation ---
    try:
        profile = UserProfile(request.user_profile.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    validation_errors = profile.validate()
    if validation_errors:
        raise HTTPException(status_code=400, detail={"errors": validation_errors})

    profile_dict = profile.to_dict()

    # --- Step 2: Session & Memory ---
    session = session_manager.get_or_create(request.session_id, profile_dict)

    # --- Step 3 & 4: Intent-Aware Retrieval ---
    chunks, retrieval_used = retrieve_context(
        message=request.message,
        user_zodiac=profile.zodiac.lower(),
        collection=knowledge_collection,
        top_k=3,
        threshold=1.5,
    )

    # --- Step 5: Build Prompt ---
    memory_context = session_manager.get_memory_context(request.session_id)

    messages = prompt_builder.build(
        profile_dict=profile_dict,
        memory_context=memory_context,
        retrieved_chunks=chunks,
        current_message=request.message,
    )

    # --- Step 6: Generate Response ---
    response_text = llm_provider.generate(messages)

    # --- Step 7: Store in Memory ---
    session_manager.add_exchange(
        session_id=request.session_id,
        user_msg=request.message,
        assistant_msg=response_text,
    )

    # --- Step 8: Build API Response ---
    context_sources = get_context_sources(chunks) if chunks else []
    turn_number = session["memory"].turn_count

    elapsed = time.time() - start_time
    print(
        f"[CHAT] session={request.session_id} | turn={turn_number} | "
        f"retrieval={retrieval_used} | sources={context_sources} | "
        f"time={elapsed:.2f}s"
    )

    return ChatResponse(
        response=response_text,
        zodiac=profile.zodiac,
        context_used=context_sources,
        retrieval_used=retrieval_used,
        session_id=request.session_id,
        turn_number=turn_number,
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "llm_provider": type(llm_provider).__name__ if llm_provider else "not initialized",
        "knowledge_base": "loaded" if knowledge_collection else "not loaded",
    }


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Astro Conversational Insight Agent",
        "version": "1.0.0",
        "endpoints": {
            "POST /chat": "Send a message to the astrology agent",
            "GET /health": "Check service health",
        },
        "docs": "/docs",
    }
