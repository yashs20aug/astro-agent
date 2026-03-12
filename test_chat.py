"""
Simple test client — run this while the server is running.

Usage:
    python test_chat.py
"""

import requests

URL = "http://localhost:8000/chat"

# User profile (same for all turns)
profile = {
    "name": "Ritika",
    "birth_date": "1995-08-20",
    "birth_time": "14:30",
    "birth_place": "Jaipur, India",
    "preferred_language": "en"
}

def chat(message, session_id="demo-session"):
    """Send a message and print the response."""
    response = requests.post(URL, json={
        "session_id": session_id,
        "message": message,
        "user_profile": profile,
    })
    data = response.json()

    print(f"\n{'='*60}")
    print(f"YOU:  {message}")
    print(f"{'─'*60}")
    print(f"ASTRO: {data['response']}")
    print(f"{'─'*60}")
    print(f"Zodiac: {data['zodiac']}  |  Retrieval: {data['retrieval_used']}  |  Sources: {data['context_used']}  |  Turn: {data['turn_number']}")
    print(f"{'='*60}")
    return data


if __name__ == "__main__":
    print("Astro Agent — Test Conversation")
    print("Keep your server running in another terminal!\n")

    # Turn 1: Career question → should trigger retrieval
    chat("How will my career be this month?")

    # Turn 2: Love question → should trigger retrieval
    chat("What about my love life?")

    # Turn 3: Summary → should SKIP retrieval
    chat("Summarize what you've told me so far")

    # Turn 4: Spiritual → should trigger retrieval
    chat("Guide me on my spiritual path")

    # Turn 5: Greeting → should SKIP retrieval
    chat("Thank you!")
