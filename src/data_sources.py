from __future__ import annotations

from datetime import timedelta
from typing import Any

import feedparser
import requests

from src.config import RSS_FEEDS
from src.utils import parse_datetime, utc_now


NEWSAPI_URL = "https://newsapi.org/v2/everything"


def _clean_text(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\n", " ").split())


def fetch_rss_events(lookback_hours: int = 24, max_events: int = 200) -> list[dict[str, Any]]:
    cutoff = utc_now() - timedelta(hours=lookback_hours)
    events: list[dict[str, Any]] = []

    for url in RSS_FEEDS:
        parsed = feedparser.parse(url)
        for entry in parsed.entries:
            published = (
                parse_datetime(entry.get("published"))
                or parse_datetime(entry.get("updated"))
                or utc_now()
            )
            if published < cutoff:
                continue

            events.append(
                {
                    "title": _clean_text(entry.get("title")),
                    "summary": _clean_text(entry.get("summary") or entry.get("description")),
                    "url": entry.get("link", ""),
                    "source": parsed.feed.get("title", url),
                    "published_utc": published,
                }
            )

    events.sort(key=lambda x: x["published_utc"], reverse=True)
    return events[:max_events]


def fetch_newsapi_events(api_key: str, lookback_hours: int = 24, max_events: int = 200) -> list[dict[str, Any]]:
    cutoff = utc_now() - timedelta(hours=lookback_hours)
    params = {
        "q": "(stocks OR earnings OR fed OR inflation OR guidance OR merger)",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(max_events, 100),
        "apiKey": api_key,
    }

    response = requests.get(NEWSAPI_URL, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()

    events: list[dict[str, Any]] = []
    for article in payload.get("articles", []):
        published = parse_datetime(article.get("publishedAt")) or utc_now()
        if published < cutoff:
            continue

        events.append(
            {
                "title": _clean_text(article.get("title")),
                "summary": _clean_text(article.get("description") or article.get("content")),
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", "NewsAPI"),
                "published_utc": published,
            }
        )

    return events


def collect_hot_events(lookback_hours: int = 24, newsapi_key: str | None = None) -> list[dict[str, Any]]:
    events = fetch_rss_events(lookback_hours=lookback_hours)

    if newsapi_key:
        try:
            events.extend(fetch_newsapi_events(newsapi_key, lookback_hours=lookback_hours))
        except Exception:
            # Keep app usable when API quota/key is unavailable.
            pass

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda x: x["published_utc"], reverse=True):
        key = (event.get("title", "").lower(), event.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)

    return deduped
