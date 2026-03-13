"""
retry_handler.py — Shared Retry Logic for Groq API Calls
==========================================================
This module provides a robust retry wrapper for all Groq LLM API calls
used across the AI system.

Why retry logic is needed:
    - LLM APIs can occasionally fail due to rate limits, timeouts,
      or temporary server errors
    - Without retries, a single failed API call would break the whole flow
    - This module automatically retries failed calls with a delay,
      and returns a safe fallback if all retries fail

Used by:
    - question_generator
    - answer_evaluator
    - ats_scorer

Features:
    - Configurable number of retries (default: 3)
    - Exponential backoff between retries (1s, 2s, 4s...)
    - Timeout detection
    - Clean fallback values on total failure
    - Detailed logging for debugging

For backend integration:
    - This module is internal — backend does NOT call this directly
    - All LLM calls in the AI modules go through call_with_retry()

Dependencies:
    pip install groq
"""

import time
import json
import re
from groq import Groq

# ─────────────────────────────────────────────
# SECTION: Retry Configuration
# ─────────────────────────────────────────────

DEFAULT_MAX_RETRIES   = 3      # Number of retry attempts before giving up
DEFAULT_RETRY_DELAY   = 1.0    # Initial delay in seconds between retries
DEFAULT_BACKOFF_FACTOR = 2.0   # Multiply delay by this after each retry
                                # Retry delays: 1s → 2s → 4s



# Core Retry Wrapper

def call_with_retry(
    client: Groq,
    messages: list,
    model: str,
    temperature: float = 0.5,
    max_tokens: int = 1024,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    caller_label: str = "LLMCall",
) -> str | None:
    """
    Sends a message to the Groq API with automatic retry on failure.
    This is the SINGLE function all modules should use for LLM API calls.

    Parameters:
        client (Groq):
            An authenticated Groq client from shared/groq_client.py.

        messages (list):
            The conversation messages in OpenAI format.
            Example:
            [
                {"role": "system", "content": "You are an expert..."},
                {"role": "user",   "content": "Generate questions..."}
            ]

        model (str):
            The Groq model to use.
            Example: "llama-3.3-70b-versatile"
            Use MODEL_CONFIGS from groq_client.py for consistent settings.

        temperature (float):
            Controls response creativity. Range: 0.0 (precise) to 1.0 (creative).
            Default: 0.5

        max_tokens (int):
            Maximum tokens in the response. Default: 1024.

        max_retries (int):
            Number of retry attempts before giving up. Default: 3.

        retry_delay (float):
            Initial wait time in seconds before first retry. Default: 1.0.

        backoff_factor (float):
            Multiplier for delay after each retry. Default: 2.0.
            Example: delay=1s → 2s → 4s

        caller_label (str):
            A label for logging to identify which module made the call.
            Example: "QuestionGenerator", "ATSScorer"

    Returns:
        str | None:
            The raw text response from the LLM if successful.
            Returns None if all retries are exhausted.

    Example usage (inside any AI module):
        from shared.groq_client import get_groq_client, MODEL_CONFIGS
        from shared.retry_handler import call_with_retry

        client = get_groq_client()
        config = MODEL_CONFIGS["question_generator"]

        response_text = call_with_retry(
            client      = client,
            messages    = [
                {"role": "system", "content": "You are an expert interviewer."},
                {"role": "user",   "content": prompt}
            ],
            model       = config["model"],
            temperature = config["temperature"],
            max_tokens  = config["max_tokens"],
            caller_label = "QuestionGenerator"
        )

        if response_text:
            questions = json.loads(response_text)
        else:
            print("All retries failed.")
    """

    attempt      = 0
    current_delay = retry_delay

    while attempt < max_retries:
        attempt += 1
        try:
            print(f"[{caller_label}] API call attempt {attempt}/{max_retries}...")

            response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                temperature = temperature,
                max_tokens  = max_tokens,
            )

            raw_text = response.choices[0].message.content.strip()

            # ── Clean markdown fences if present ──
            raw_text = re.sub(r"```json|```", "", raw_text).strip()

            print(f"[{caller_label}] API call succeeded on attempt {attempt}.")
            return raw_text

        except Exception as e:
            error_message = str(e)

            # ── Detect rate limit errors ──
            if "rate_limit" in error_message.lower() or "429" in error_message:
                print(f"[{caller_label}] Rate limit hit. Waiting {current_delay}s before retry...")

            # ── Detect model decommissioned error ──
            elif "decommissioned" in error_message.lower():
                print(f"[{caller_label}] ERROR: Model is decommissioned. Update the model name in shared/groq_client.py")
                return None  # No point retrying if model is invalid

            # ── General error ──
            else:
                print(f"[{caller_label}] Attempt {attempt} failed: {error_message}")

            # ── Wait before retrying (unless it's the last attempt) ──
            if attempt < max_retries:
                print(f"[{caller_label}] Retrying in {current_delay}s...")
                time.sleep(current_delay)
                current_delay *= backoff_factor  # Exponential backoff

    print(f"[{caller_label}] All {max_retries} attempts failed. Returning None.")
    return None


# JSON Parse Helper

def parse_json_response(raw_text: str | None, caller_label: str = "LLMCall") -> dict | list | None:
    """
    Safely parses a JSON string returned by the LLM.
    Handles common issues like extra whitespace or stray characters.

    Parameters:
        raw_text (str | None):  Raw text response from call_with_retry().
                                If None, returns None immediately.
        caller_label (str):     Label for logging.

    Returns:
        dict | list | None:
            Parsed Python object if successful.
            Returns None if parsing fails.

    Example usage:
        raw = call_with_retry(client, messages, model, ...)
        result = parse_json_response(raw, "QuestionGenerator")
        if result:
            questions = result  # list or dict
    """
    if raw_text is None:
        print(f"[{caller_label}] No response to parse (raw_text is None).")
        return None

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        print(f"[{caller_label}] JSON parse error: {e}")
        print(f"[{caller_label}] Raw text was: {raw_text[:200]}...")
        return None


# Test

if __name__ == "__main__":
    """
    Quick test — verifies retry handler works with a simple API call.

    Usage (from project root):
        python -m shared.retry_handler
    """
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    try:
        from shared.groq_client import get_groq_client, MODEL_CONFIGS
    except ImportError:
        from groq_client import get_groq_client, MODEL_CONFIGS

    print("[RetryHandler] Testing retry handler with a simple API call...")

    client = get_groq_client()
    config = MODEL_CONFIGS["question_generator"]

    response = call_with_retry(
        client       = client,
        messages     = [
            {"role": "system", "content": "You are a helpful assistant. Always respond with only valid JSON."},
            {"role": "user",   "content": 'Respond with this exact JSON: {"status": "ok", "message": "retry handler works"}'}
        ],
        model        = config["model"],
        temperature  = config["temperature"],
        max_tokens   = 64,
        caller_label = "RetryHandlerTest",
    )

    result = parse_json_response(response, "RetryHandlerTest")

    if result:
        print(f"[RetryHandler] SUCCESS! Response: {result}")
    else:
        print("[RetryHandler] FAILED: Could not get a valid response.")