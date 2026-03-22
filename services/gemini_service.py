"""
Verification Service (Tavily + Groq)
Uses Tavily for web evidence retrieval and Groq for final fact-check reasoning.
Keeps verify_news(...) return format unchanged for existing routes/storage.
"""

import json
import os
from typing import List, Tuple

import requests
from tavily import TavilyClient

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_FALLBACK = {
    "verdict": "Uncertain",
    "explanation": "LLM verification could not be completed.",
    "sources": "",
}


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is not set in .env")
    return value


def _safe_json_loads(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return json.loads(cleaned)


def _search_web_evidence(news_text: str) -> Tuple[str, List[str]]:
    tavily_key = _required_env("TAVILY_API_KEY")
    client = TavilyClient(tavily_key)
    result = client.search(
        query=news_text,
        topic="news",
        search_depth="advanced",
        max_results=5,
    )

    evidence_lines: List[str] = []
    urls: List[str] = []
    for i, item in enumerate(result.get("results", [])[:5], start=1):
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        url = str(item.get("url", "")).strip()
        evidence_lines.append(f"{i}. Title: {title}\nSnippet: {content}")
        if url:
            urls.append(url)

    return "\n\n".join(evidence_lines), urls


def _verify_with_groq(news_text: str, evidence_text: str) -> dict:
    groq_key = _required_env("GROQ_API_KEY")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    prompt = (
        "You are a fact-checking assistant for Urdu news.\n"
        "Use the web evidence to classify the claim as Real, Fake, or Uncertain.\n"
        "Rules:\n"
        "1. If evidence is conflicting or weak, return Uncertain.\n"
        "2. If major claim contradicts reliable reporting, return Fake.\n"
        "3. If claim is well supported by evidence, return Real.\n"
        "4. Return JSON only, no markdown and no extra text.\n\n"
        "JSON schema:\n"
        '{"verdict":"Real|Fake|Uncertain","explanation":"2-3 short sentences in English",'
        '"sources":"comma-separated source URLs"}\n\n'
        f"News text:\n{news_text}\n\n"
        f"Web evidence:\n{evidence_text}\n"
    )

    payload = {
        "model": groq_model,
        "temperature": 0.1,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    }
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return _safe_json_loads(content)


def verify_news(news_text: str) -> dict:
    """
    Verify whether the given Urdu news text is real/fake using Tavily + Groq.

    Returns: {
        "verdict": "Real" | "Fake" | "Uncertain",
        "explanation": "...",
        "sources": "..."
    }
    """
    try:
        evidence_text, urls = _search_web_evidence(news_text)
        parsed = _verify_with_groq(news_text, evidence_text)

        verdict = str(parsed.get("verdict", "Uncertain")).strip()
        if verdict not in ("Real", "Fake", "Uncertain"):
            verdict = "Uncertain"

        sources = str(parsed.get("sources", "")).strip()
        if not sources and urls:
            sources = ", ".join(urls)

        return {
            "verdict": verdict,
            "explanation": str(parsed.get("explanation", "")).strip(),
            "sources": sources,
        }
    except (requests.RequestException, json.JSONDecodeError, KeyError, IndexError, RuntimeError) as e:
        print(f"[verify] Tavily+Groq verification failed: {e}")
        return _FALLBACK
