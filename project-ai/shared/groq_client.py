"""
groq_client.py — Shared Groq API Client
========================================
This module provides a single, shared Groq API client used by
ALL modules in the AI system:
    - question_generator
    - answer_evaluator
    - ats_scorer

Instead of each module creating its own Groq client, they all
import from here. This means:
    - API key is loaded in ONE place only
    - Any client configuration change applies everywhere
    - Easier to maintain and debug

For backend integration:
    - This module is internal — backend does NOT call this directly
    - It is used automatically by question_generator, answer_evaluator, ats_scorer

Dependencies:
    pip install groq python-dotenv
"""

import os
from groq import Groq
from dotenv import load_dotenv

# Load .env file so GROQ_API_KEY is available
load_dotenv()

# Groq Client (Singleton Pattern)

# Module-level client instance — created once and reused
# This avoids creating a new client on every function call
_groq_client = None


def get_groq_client() -> Groq:
    """
    Returns a shared Groq API client instance.
    Creates the client only once (singleton pattern) and reuses it.

    This function is called by question_generator, answer_evaluator,
    and ats_scorer — they should all import from here instead of
    creating their own clients.

    Returns:
        Groq: An authenticated Groq client instance.

    Raises:
        ValueError: If GROQ_API_KEY is not found in the .env file.

    Example usage (inside any AI module):
        from shared.groq_client import get_groq_client
        client = get_groq_client()
    """
    global _groq_client

    # Return existing client if already created
    if _groq_client is not None:
        return _groq_client

    # Load API key from environment
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "[GroqClient] ERROR: GROQ_API_KEY not found.\n"
            "Please add it to your .env file:\n"
            "    GROQ_API_KEY=your_key_here\n"
            "Get your free API key at: https://console.groq.com"
        )

    # Create and cache the client
    _groq_client = Groq(api_key=api_key)
    print("[GroqClient] Groq client initialized successfully.")

    return _groq_client


def reset_groq_client() -> None:
    """
    Resets the cached Groq client instance.
    Useful for testing or when the API key changes at runtime.

    Returns:
        None

    Example usage:
        from shared.groq_client import reset_groq_client
        reset_groq_client()  # Forces re-initialization on next call
    """
    global _groq_client
    _groq_client = None
    print("[GroqClient] Groq client reset.")



# Model Configuration

# Default model used across all modules
# Change this ONE value to switch models everywhere
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Model settings per use case
# Modules can import these instead of hardcoding values
MODEL_CONFIGS = {
    "question_generator": {
        "model":       DEFAULT_MODEL,
        "temperature": 0.7,     # Higher creativity for varied questions
        "max_tokens":  1024,
    },
    "answer_evaluator": {
        "model":       DEFAULT_MODEL,
        "temperature": 0.3,     # Lower temperature for consistent evaluation
        "max_tokens":  1024,
    },
    "ats_scorer": {
        "model":       DEFAULT_MODEL,
        "temperature": 0.3,     # Lower temperature for consistent scoring
        "max_tokens":  256,
    },
}