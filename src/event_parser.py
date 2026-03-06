from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from typing import Any

import requests

from src.config import EVENT_THEMES, NEGATIVE_WORDS, POSITIVE_WORDS, TICKER_ALIASES
from src.utils import clip, utc_now

TICKER_PATTERN = re.compile(r"\$?\b[A-Z]{1,5}(?:-[A-Z])?\b")
FLOAT_PATTERN = re.compile(r"[-+]?\d*\.?\d+")

_GPT_SENTIMENT_CACHE: dict[tuple[str, str], float] = {}
_GPT_CACHE_LIMIT = 500


def _new_sentiment_stats() -> dict[str, Any]:
    return {
        "gpt_requested_events": 0,
        "gpt_requested_without_key": 0,
        "gpt_api_calls": 0,
        "gpt_success_events": 0,
        "gpt_failed_events": 0,
        "gpt_cache_hits": 0,
        "heuristic_events": 0,
        "heuristic_fallback_events": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "last_gpt_error": "",
    }


_SENTIMENT_STATS = _new_sentiment_stats()


def reset_sentiment_stats() -> None:
    global _SENTIMENT_STATS
    _SENTIMENT_STATS = _new_sentiment_stats()


def get_sentiment_stats() -> dict[str, Any]:
    return dict(_SENTIMENT_STATS)


def _inc_stat(name: str, amount: int = 1) -> None:
    _SENTIMENT_STATS[name] = int(_SENTIMENT_STATS.get(name, 0) or 0) + amount


def _set_last_error(exc: Exception) -> None:
    _SENTIMENT_STATS["last_gpt_error"] = str(exc)[:300]


def _add_usage(usage: dict[str, Any] | None) -> None:
    if not isinstance(usage, dict):
        return

    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    total = int(usage.get("total_tokens", prompt + completion) or 0)

    _inc_stat("prompt_tokens", prompt)
    _inc_stat("completion_tokens", completion)
    _inc_stat("total_tokens", total)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9&']+", text.lower())


def infer_theme(text: str) -> str:
    lowered = text.lower()
    best_theme = "general"
    best_hits = 0
    for theme, words in EVENT_THEMES.items():
        hits = sum(1 for word in words if word in lowered)
        if hits > best_hits:
            best_theme = theme
            best_hits = hits
    return best_theme


def _heuristic_sentiment_score(text: str) -> float:
    words = _tokenize(text)
    if not words:
        return 0.0

    counts = Counter(words)
    pos = sum(counts[word] for word in POSITIVE_WORDS if word in counts)
    neg = sum(counts[word] for word in NEGATIVE_WORDS if word in counts)
    raw = (pos - neg) / max(4, len(words) ** 0.5)
    return math.tanh(raw)


def _extract_response_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        return ""

    message = choices[0].get("message", {})
    content = message.get("content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text") or item.get("value") or ""
                if text_value:
                    text_parts.append(str(text_value))
        return " ".join(text_parts)

    return str(content)


def _gpt_sentiment_score(
    text: str,
    openai_api_key: str,
    openai_model: str = "gpt-4o-mini",
    timeout_sec: int = 25,
) -> tuple[float, dict[str, Any] | None, bool]:
    trimmed_text = text.strip()
    if not trimmed_text:
        return 0.0, None, False

    cache_key = (openai_model, trimmed_text)
    cached = _GPT_SENTIMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached, None, True

    prompt = (
        "Score the sentiment impact of this market news for the stock's NEXT U.S. trading day move. "
        "Return only one number between -1 and 1. "
        "-1 = strongly bearish, 0 = neutral/unclear, 1 = strongly bullish. "
        "No explanation, no extra text."
    )

    payload = {
        "model": openai_model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": trimmed_text[:4000]},
        ],
        "max_tokens": 10,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout_sec,
    )
    response.raise_for_status()

    response_json = response.json()
    content = _extract_response_text(response_json)
    match = FLOAT_PATTERN.search(content)
    if not match:
        raise ValueError("No numeric sentiment returned by GPT model")

    score = clip(float(match.group(0)), -1.0, 1.0)

    if len(_GPT_SENTIMENT_CACHE) >= _GPT_CACHE_LIMIT:
        _GPT_SENTIMENT_CACHE.clear()
    _GPT_SENTIMENT_CACHE[cache_key] = score

    usage = response_json.get("usage") if isinstance(response_json, dict) else None
    return score, usage, False


def sentiment_score(
    text: str,
    use_gpt_sentiment: bool = False,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4o-mini",
) -> float:
    if use_gpt_sentiment:
        _inc_stat("gpt_requested_events")

        if not openai_api_key:
            _inc_stat("gpt_requested_without_key")
            _inc_stat("heuristic_events")
            return _heuristic_sentiment_score(text)

        try:
            score, usage, cache_hit = _gpt_sentiment_score(
                text=text,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )
            _inc_stat("gpt_success_events")
            if cache_hit:
                _inc_stat("gpt_cache_hits")
            else:
                _inc_stat("gpt_api_calls")
            _add_usage(usage)
            return score
        except Exception as exc:
            _inc_stat("gpt_failed_events")
            _inc_stat("heuristic_fallback_events")
            _set_last_error(exc)
            return _heuristic_sentiment_score(text)

    _inc_stat("heuristic_events")
    return _heuristic_sentiment_score(text)


def extract_tickers(text: str) -> list[str]:
    tickers: set[str] = set()

    # Capture explicit ticker-like mentions: TSLA, $TSLA, BRK-B.
    for token in TICKER_PATTERN.findall(text.upper()):
        if token in TICKER_ALIASES:
            tickers.add(token)

    lowered = text.lower()
    for ticker, aliases in TICKER_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                tickers.add(ticker)
                break

    return sorted(tickers)


def event_to_signals(
    event: dict[str, Any],
    now_utc: datetime | None = None,
    use_gpt_sentiment: bool = False,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4o-mini",
) -> list[dict[str, Any]]:
    text = f"{event.get('title', '')}. {event.get('summary', '')}".strip()
    tickers = extract_tickers(text)
    if not tickers:
        return []

    score = sentiment_score(
        text=text,
        use_gpt_sentiment=use_gpt_sentiment,
        openai_api_key=openai_api_key,
        openai_model=openai_model,
    )
    theme = infer_theme(text)
    published = event.get("published_utc") or utc_now()
    current = now_utc or utc_now()
    age_hours = max(0.0, (current - published).total_seconds() / 3600)
    recency_weight = math.exp(-age_hours / 18)

    signals: list[dict[str, Any]] = []
    for ticker in tickers:
        signals.append(
            {
                "ticker": ticker,
                "event_score": score,
                "recency_weight": recency_weight,
                "weighted_event_score": score * recency_weight,
                "theme": theme,
                "headline": event.get("title", ""),
                "url": event.get("url", ""),
                "source": event.get("source", ""),
                "published_utc": published,
            }
        )

    return signals
