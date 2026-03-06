from __future__ import annotations

from datetime import date, datetime
from typing import Any

import pandas as pd
import requests


def _safe_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _to_utc_timestamp(value: Any) -> str:
    if value is None:
        return "Unknown time"

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M UTC")

    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            return "Unknown time"
        return parsed.to_pydatetime().strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return "Unknown time"


def _collect_ticker_news(signals_df: pd.DataFrame, ticker: str, max_items: int = 4) -> list[dict[str, str]]:
    if signals_df.empty or "ticker" not in signals_df.columns:
        return []

    df = signals_df[signals_df["ticker"] == ticker].copy()
    if df.empty:
        return []

    if "published_utc" in df.columns:
        df = df.sort_values("published_utc", ascending=False)

    items: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        headline = str(row.get("headline", "")).strip()
        url = str(row.get("url", "")).strip()
        source = str(row.get("source", "")).strip()
        ts = _to_utc_timestamp(row.get("published_utc"))

        if not headline:
            continue

        key = (headline.lower(), url)
        if key in seen:
            continue
        seen.add(key)

        items.append(
            {
                "headline": headline,
                "url": url,
                "source": source,
                "published_utc": ts,
            }
        )

        if len(items) >= max_items:
            break

    return items


def _format_news_references(news_items: list[dict[str, str]]) -> str:
    if not news_items:
        return "**Referenced News (UTC):**\n- No ticker-specific headline was matched in this run."

    lines = ["**Referenced News (UTC):**"]
    for item in news_items:
        line = f"- [{item['published_utc']}] {item['headline']}"
        if item.get("source"):
            line += f" ({item['source']})"
        if item.get("url"):
            line += f" - {item['url']}"
        lines.append(line)
    return "\n".join(lines)


def _heuristic_summary(row: pd.Series, ticker: str, target_day: date, news_items: list[dict[str, str]]) -> str:
    themes = str(row.get("themes", "general")).strip() or "general"
    event_count = int(_safe_float(row, "event_count", 0))
    sentiment = _safe_float(row, "raw_event_sentiment", 0.0)
    ret_1d = _safe_float(row, "ret_1d", 0.0)
    ret_5d = _safe_float(row, "ret_5d", 0.0)
    ret_20d = _safe_float(row, "ret_20d", 0.0)
    vol_ratio = _safe_float(row, "vol_ratio", 1.0)

    if sentiment > 0.15:
        tone = "net supportive"
    elif sentiment < -0.15:
        tone = "net cautious"
    else:
        tone = "mixed"

    if ret_1d > 0 and ret_5d > 0 and ret_20d > 0:
        trend = "trend structure has been constructive across short and medium windows"
    elif ret_1d < 0 and ret_5d < 0 and ret_20d < 0:
        trend = "recent trend has been weak, so confirmation after the open is important"
    else:
        trend = "trend structure is mixed, making intraday confirmation important"

    if vol_ratio >= 1.25:
        participation = "participation is elevated, indicating stronger market attention"
    elif vol_ratio <= 0.85:
        participation = "participation is light, so follow-through risk is higher"
    else:
        participation = "participation is near normal, so headline quality matters most"

    summary = (
        f"For {target_day:%A, %B %d, %Y}, {ticker} is included because recent event flow and market context align "
        f"for a short-horizon setup. The dominant event themes are {themes}, with {event_count} relevant headline "
        f"matches and a {tone} aggregate sentiment tone. The thesis is that these developments can influence next-session "
        f"expectations quickly, especially when they affect guidance outlook, demand assumptions, regulatory interpretation, "
        f"or strategic execution confidence. On tape context, {trend}. In addition, {participation}. Together, this favors "
        f"{ticker} as a tactical next-day candidate rather than a long-duration view. Execution should still be conditional: "
        f"if pre-market and opening-hour price action confirms the direction implied by the latest headlines, conviction improves; "
        f"if price reaction diverges materially from headline tone, risk should be reduced quickly."
    )

    return summary


def _build_gpt_prompt(
    ticker: str,
    target_day: date,
    row: pd.Series,
    news_items: list[dict[str, str]],
) -> str:
    themes = str(row.get("themes", "general")).strip() or "general"
    event_count = int(_safe_float(row, "event_count", 0))
    sentiment = _safe_float(row, "raw_event_sentiment", 0.0)
    ret_1d = _safe_float(row, "ret_1d", 0.0)
    ret_5d = _safe_float(row, "ret_5d", 0.0)
    ret_20d = _safe_float(row, "ret_20d", 0.0)
    vol_ratio = _safe_float(row, "vol_ratio", 1.0)

    news_lines = []
    for item in news_items:
        news_lines.append(
            f"- [{item['published_utc']}] {item['headline']} (source: {item.get('source','unknown')})"
        )

    if not news_lines:
        news_lines.append("- No ticker-specific headline matched in this run.")

    return (
        f"Write a concise investment rationale (170-230 words) for why {ticker} is recommended for "
        f"the next U.S. trading session on {target_day:%A, %B %d, %Y}.\n\n"
        "Requirements:\n"
        "1) Focus on reasoning, headline implications, and next-day trade logic.\n"
        "2) Do NOT repeat dashboard metrics like rank, score, or confidence.\n"
        "3) You may use event context and market context from inputs below.\n"
        "4) If you mention any specific news headline, include its exact UTC timestamp in square brackets, like [YYYY-MM-DD HH:MM UTC].\n"
        "5) Use only provided news lines for citations.\n\n"
        "Inputs:\n"
        f"- Themes: {themes}\n"
        f"- Matched headline count: {event_count}\n"
        f"- Average sentiment score (for context only): {sentiment:.3f}\n"
        f"- Recent returns: 1d={ret_1d:.4f}, 5d={ret_5d:.4f}, 20d={ret_20d:.4f}\n"
        f"- Volume ratio vs 20d avg: {vol_ratio:.3f}\n"
        "- News lines:\n"
        + "\n".join(news_lines)
    )


def _gpt_summary(
    ticker: str,
    target_day: date,
    row: pd.Series,
    news_items: list[dict[str, str]],
    openai_api_key: str,
    openai_model: str,
    timeout_sec: int = 35,
) -> str:
    prompt = _build_gpt_prompt(ticker, target_day, row, news_items)

    payload = {
        "model": openai_model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a market analyst writing concise, evidence-grounded next-day trade rationales. "
                    "Do not fabricate headlines or timestamps."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 350,
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
    data = response.json()

    choices = data.get("choices", [])
    if not choices:
        raise ValueError("No choices returned for GPT summary")

    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        text = " ".join(parts).strip()
    else:
        text = str(content).strip()

    if not text:
        raise ValueError("Empty GPT summary output")

    return text


def build_recommendation_summaries(
    rec_df: pd.DataFrame,
    signals_df: pd.DataFrame,
    target_day: date,
    use_gpt_summary: bool,
    openai_api_key: str | None,
    openai_model: str,
) -> dict[str, dict[str, str]]:
    summaries: dict[str, dict[str, str]] = {}
    if rec_df.empty:
        return summaries

    for _, row in rec_df.sort_values("rank").iterrows():
        ticker = str(row.get("ticker", "")).strip()
        if not ticker:
            continue

        news_items = _collect_ticker_news(signals_df, ticker, max_items=4)

        backend = "heuristic"
        error_msg = ""

        if use_gpt_summary and openai_api_key:
            try:
                body = _gpt_summary(
                    ticker=ticker,
                    target_day=target_day,
                    row=row,
                    news_items=news_items,
                    openai_api_key=openai_api_key,
                    openai_model=openai_model,
                )
                backend = "gpt"
            except Exception as exc:
                body = _heuristic_summary(row, ticker, target_day, news_items)
                backend = "heuristic_fallback"
                error_msg = str(exc)[:300]
        else:
            body = _heuristic_summary(row, ticker, target_day, news_items)

        full_text = body.strip() + "\n\n" + _format_news_references(news_items)
        summaries[ticker] = {
            "text": full_text,
            "backend": backend,
            "error": error_msg,
        }

    return summaries
