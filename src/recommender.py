from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import DEFAULT_LOOKBACK_HOURS, DEFAULT_NUM_PICKS, TICKER_ALIASES
from src.data_sources import collect_hot_events
from src.event_parser import event_to_signals, get_sentiment_stats, reset_sentiment_stats
from src.scoring import add_portfolio_weights, aggregate_recommendations, fetch_market_features
from src.utils import get_next_trading_day, utc_now


FALLBACK_MOMENTUM_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD", "AVGO", "JPM",
    "XOM", "LLY", "UNH", "V", "MA", "WMT", "COST", "PG", "KO", "PEP",
]


def build_next_day_recommendations(
    lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
    num_picks: int = DEFAULT_NUM_PICKS,
    newsapi_key: str | None = None,
    use_gpt_sentiment: bool = False,
    openai_api_key: str | None = None,
    openai_model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    generated_at = utc_now()
    events = collect_hot_events(lookback_hours=lookback_hours, newsapi_key=newsapi_key)

    reset_sentiment_stats()

    signal_rows: list[dict[str, Any]] = []
    for event in events:
        signal_rows.extend(
            event_to_signals(
                event,
                now_utc=generated_at,
                use_gpt_sentiment=use_gpt_sentiment,
                openai_api_key=openai_api_key,
                openai_model=openai_model,
            )
        )

    signals_df = pd.DataFrame(signal_rows)

    if not signals_df.empty:
        candidate_tickers = sorted(signals_df["ticker"].unique().tolist())
    else:
        candidate_tickers = FALLBACK_MOMENTUM_TICKERS

    # Keep universe bounded for runtime speed.
    candidate_tickers = [t for t in candidate_tickers if t in TICKER_ALIASES][:80]

    market_df = fetch_market_features(candidate_tickers)
    recommendations = aggregate_recommendations(signals_df, market_df, num_picks=num_picks)

    if recommendations.empty and not market_df.empty:
        recommendations = market_df.sort_values("momentum_score", ascending=False).head(num_picks).copy()
        recommendations.insert(0, "rank", range(1, len(recommendations) + 1))
        recommendations["score"] = recommendations["momentum_score"]
        recommendations["confidence"] = 0.45 + 0.2 * recommendations["score"].abs().clip(upper=1.0)
        recommendations["event_score"] = 0.0
        recommendations["raw_event_sentiment"] = 0.0
        recommendations["event_count"] = 0
        recommendations["themes"] = "momentum"
        recommendations["catalysts"] = "Momentum-only fallback"

    recommendations = add_portfolio_weights(recommendations)

    next_day = get_next_trading_day(generated_at)

    sentiment_stats = get_sentiment_stats()
    gpt_success = int(sentiment_stats.get("gpt_success_events", 0) or 0)
    gpt_failed = int(sentiment_stats.get("gpt_failed_events", 0) or 0)
    gpt_requested = int(sentiment_stats.get("gpt_requested_events", 0) or 0)
    last_gpt_error = str(sentiment_stats.get("last_gpt_error", "") or "")

    sentiment_backend = "heuristic"
    sentiment_backend_reason = "Heuristic sentiment mode selected."

    if use_gpt_sentiment and not openai_api_key:
        sentiment_backend = "heuristic_no_openai_key"
        sentiment_backend_reason = "GPT mode was requested but no OpenAI API key was provided for this run."
    elif gpt_success > 0 and gpt_failed > 0:
        sentiment_backend = "gpt_with_partial_fallback"
        sentiment_backend_reason = (
            f"GPT scored {gpt_success} event(s), but {gpt_failed} event(s) fell back to heuristic. "
            f"Latest GPT error: {last_gpt_error or 'unknown error'}."
        )
    elif gpt_success > 0:
        sentiment_backend = "gpt"
        sentiment_backend_reason = f"GPT successfully scored {gpt_success} event(s)."
    elif gpt_requested > 0 and gpt_failed > 0:
        sentiment_backend = "heuristic_due_to_gpt_errors"
        sentiment_backend_reason = (
            f"GPT was requested but all attempted GPT calls failed ({gpt_failed} event(s)); "
            f"heuristic fallback was used. Latest GPT error: {last_gpt_error or 'unknown error'}."
        )
    elif gpt_requested > 0 and gpt_success == 0:
        sentiment_backend = "heuristic_no_gpt_attempts"
        sentiment_backend_reason = (
            "GPT mode was requested, but no ticker-matched events required sentiment scoring in this run."
        )

    return {
        "generated_at_utc": generated_at,
        "next_trading_day": next_day,
        "events": events,
        "signals_df": signals_df,
        "recommendations_df": recommendations,
        "sentiment_backend": sentiment_backend,
        "sentiment_backend_reason": sentiment_backend_reason,
        "openai_model": openai_model,
        "sentiment_stats": sentiment_stats,
    }
