"""
shared/__init__.py
===================
Exposes the shared utilities used across all AI modules.

All AI modules (question_generator, answer_evaluator, ats_scorer)
should import Groq client and retry logic from here instead of
defining their own.

Available imports:
    from shared.groq_client  import get_groq_client, MODEL_CONFIGS
    from shared.retry_handler import call_with_retry, parse_json_response

Quick reference:
─────────────────────────────────────────────────────────────────
get_groq_client()
    → Returns a shared Groq client instance (singleton)
    → Reads GROQ_API_KEY from .env automatically

MODEL_CONFIGS
    → Dict of model + temperature + max_tokens per module
    → Keys: "question_generator", "answer_evaluator", "ats_scorer"

call_with_retry(client, messages, model, ...)
    → Sends an API call with automatic retry on failure
    → Returns raw text string or None on total failure

parse_json_response(raw_text, caller_label)
    → Safely parses JSON string from LLM response
    → Returns dict/list or None on parse failure
─────────────────────────────────────────────────────────────────
"""

from shared.groq_client import get_groq_client, reset_groq_client, MODEL_CONFIGS, DEFAULT_MODEL
from shared.retry_handler import call_with_retry, parse_json_response

__all__ = [
    "get_groq_client",
    "reset_groq_client",
    "MODEL_CONFIGS",
    "DEFAULT_MODEL",
    "call_with_retry",
    "parse_json_response",
]