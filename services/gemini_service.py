"""
Gemini Verification Service
Uses Google's Gemini API to verify news articles by cross-referencing
with web knowledge. Acts as a second verification layer after the ML model.
"""

import os
import json
import time
import requests

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

# ── Cooldown: skip Gemini for 60s after a rate-limit hit ─────────
_cooldown_until = 0.0
_COOLDOWN_SECONDS = 60

_FALLBACK = {
    "verdict": "Uncertain",
    "explanation": "Gemini verification could not be completed.",
    "sources": "",
}


def _get_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set in .env")
    return key


def verify_news(news_text: str) -> dict:
    """
    Ask Gemini to verify whether the given Urdu news text is real or fake
    based on its web knowledge.

    Returns: {
        "verdict": "Real" | "Fake" | "Uncertain",
        "explanation": "...",
        "sources": "..."
    }
    """
    global _cooldown_until

    # Skip immediately if we were recently rate-limited
    if time.time() < _cooldown_until:
        remaining = int(_cooldown_until - time.time())
        print(f"[gemini] Skipped — cooldown active ({remaining}s left)")
        return _FALLBACK

    api_key = _get_api_key()

    prompt = (
        "You are a fact-checking assistant. A user has submitted the following Urdu news article. "
        "Your job is to determine whether this news is REAL or FAKE based on your knowledge and "
        "information available on the web.\n\n"
        "IMPORTANT RULES:\n"
        "1. Analyze the claims made in the news text carefully.\n"
        "2. Cross-reference with known facts, events, and reliable sources.\n"
        "3. If the news contains false claims, exaggerations, or misinformation, classify it as Fake.\n"
        "4. If the news is factually accurate based on available information, classify it as Real.\n"
        "5. Respond ONLY with a valid JSON object in this exact format (no markdown, no code fences):\n"
        '{"verdict": "Real" or "Fake", "explanation": "Brief explanation in English (2-3 sentences)", '
        '"sources": "Mention any known sources or events that support your verdict"}\n\n'
        f"NEWS TEXT:\n{news_text}\n\n"
        "JSON RESPONSE:"
    )

    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 512,
        },
    }

    try:
        retries = 2
        for attempt in range(retries):
            response = requests.post(
                f"{GEMINI_API_URL}?key={api_key}",
                json=payload,
                timeout=30,
            )
            if response.status_code == 429:
                if attempt < retries - 1:
                    print(f"[gemini] Rate limited (429). Retrying in 5s (attempt {attempt + 1}/{retries})...")
                    time.sleep(5)
                    continue
                # All retries exhausted — activate cooldown
                _cooldown_until = time.time() + _COOLDOWN_SECONDS
                print(f"[gemini] Rate limited — entering {_COOLDOWN_SECONDS}s cooldown")
                return _FALLBACK
            response.raise_for_status()
            break

        data = response.json()
        text_response = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )

        # Clean up markdown code fences if Gemini wraps the response
        if text_response.startswith("```"):
            text_response = text_response.strip("`").strip()
            if text_response.startswith("json"):
                text_response = text_response[4:].strip()

        parsed = json.loads(text_response)

        verdict = parsed.get("verdict", "").strip()
        if verdict not in ("Real", "Fake"):
            verdict = "Fake"  # fail-safe: treat uncertain as fake

        return {
            "verdict": verdict,
            "explanation": parsed.get("explanation", ""),
            "sources": parsed.get("sources", ""),
        }

    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"[gemini] Verification failed: {e}")
        return _FALLBACK
