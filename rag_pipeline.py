"""
rag_pipeline.py
Generates answers using the Groq free API.
Includes context truncation to stay within the 6000 TPM free tier limit.
"""

import logging
import requests
from config import GROQ_API_KEY, GROQ_MODEL, MAX_NEW_TOKENS, TEMPERATURE

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Free tier limit is 6000 tokens per minute.
# We reserve ~500 for the question + system prompt + answer.
# Rough rule: 1 token ≈ 4 characters
MAX_CONTEXT_CHARS = 4000 * 4  # ~4000 tokens worth of context


def _truncate_context(context: str) -> str:
    """Trim context to fit within the free tier token limit."""
    if len(context) <= MAX_CONTEXT_CHARS:
        return context
    truncated = context[:MAX_CONTEXT_CHARS]
    # Cut at last newline so we don't break mid-field
    last_newline = truncated.rfind("\n")
    if last_newline > MAX_CONTEXT_CHARS // 2:
        truncated = truncated[:last_newline]
    logger.warning(
        "Context truncated from %d to %d chars to fit token limit",
        len(context), len(truncated)
    )
    return truncated + "\n[... context trimmed to fit token limit ...]"


def generate_answer(context, question):
    """Call the Groq API and return the generated answer."""

    safe_context = _truncate_context(context)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant analyzing event/incident data. "
                    "Answer ONLY using the context provided. "
                    "Be concise and specific. "
                    "If the answer is not in the context, say "
                    "'I don't have enough information to answer that.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{safe_context}\n\nQuestion: {question}",
            },
        ],
        "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

    except requests.exceptions.HTTPError as exc:
        logger.error("Groq API HTTP error: %s | Response: %s", exc, response.text)
        if response.status_code == 401:
            return "Error: Invalid Groq API key. Check GROQ_API_KEY in your .env file."
        if response.status_code == 429:
            return "Rate limit reached. Please wait a moment and try again."
        if response.status_code == 413:
            return "Context still too large. Please try a more specific question."
        return f"Groq API error ({response.status_code}): {response.text}"

    except requests.exceptions.RequestException as exc:
        logger.error("Network error: %s", exc)
        return f"Network error: {exc}"


def test_llm_connection():
    """Return True if Groq API responds successfully."""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5,
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=10)
        return response.status_code == 200
    except Exception:
        return False