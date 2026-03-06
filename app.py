from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.config import DEFAULT_LOOKBACK_HOURS, DEFAULT_NUM_PICKS
from src.explanations import build_recommendation_summaries
from src.recommender import build_next_day_recommendations

load_dotenv()

# Estimated pricing per 1M tokens (USD) from OpenAI API pricing page as of 2026-03-06.
MODEL_PRICING_PER_1M = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}

SENTIMENT_MODE_OPTIONS = [
    "Auto (GPT if key provided)",
    "GPT only (require key)",
    "Heuristic only",
]


def _resolve_pricing(model_name: str) -> dict[str, float] | None:
    if model_name in MODEL_PRICING_PER_1M:
        return MODEL_PRICING_PER_1M[model_name]

    for canonical, prices in MODEL_PRICING_PER_1M.items():
        if model_name.startswith(canonical):
            return prices

    return None


def _estimate_cost_usd(sentiment_stats: dict[str, int], model_name: str) -> float | None:
    pricing = _resolve_pricing(model_name)
    if pricing is None:
        return None

    prompt_tokens = int(sentiment_stats.get("prompt_tokens", 0) or 0)
    completion_tokens = int(sentiment_stats.get("completion_tokens", 0) or 0)

    cost_input = (prompt_tokens / 1_000_000) * pricing["input"]
    cost_output = (completion_tokens / 1_000_000) * pricing["output"]
    return cost_input + cost_output


def _detect_key_source(input_key: str, env_key: str) -> str:
    if not input_key:
        return "none"

    if not env_key:
        return "dashboard input"

    if input_key == env_key:
        return ".env (OPENAI_API_KEY)"

    return "dashboard input (overrides .env)"


def _render_methodology() -> None:
    st.subheader("Methodology")
    with st.expander("Scoring Formula", expanded=False):
        st.markdown("`score = 0.65 * event_score + 0.25 * momentum_score + 0.10 * vol_score`")
        st.markdown(
            "`event_score = sum(sentiment * exp(-age_hours / 18))` over matched hot events per ticker"
        )
        st.markdown(
            "`sentiment` is in `[-1, 1]` (GPT or heuristic), with higher values indicating more bullish event tone"
        )
        st.markdown(
            "`momentum_score = clip(2.0*ret_1d + 2.5*ret_5d + 1.0*ret_20d, -1, 1)`"
        )
        st.markdown(
            "`vol_score = clip((vol_ratio - 1.0)/1.5, -1, 1)`, where "
            "`vol_ratio = latest_volume / avg_20d_volume`"
        )
        st.markdown(
            "`confidence = clip(0.45 + 0.10*min(event_count,4) + 0.35*min(abs(score),1.0), 0, 0.99)`"
        )
        st.markdown(
            "`portfolio_signal = ((score - min(score)) + 0.05) * (0.5 + confidence)` for selected picks"
        )
        st.markdown("`portfolio_weight = portfolio_signal / sum(portfolio_signal)` (long-only, weights sum to 100%)")
        st.caption("Event sentiment backend is either heuristic keywords or GPT (if enabled and key available).")


st.set_page_config(page_title="Next-Day Stock Recommender", layout="wide")
st.title("Next-Day Stock Recommender")
st.caption(
    "Collects recent hot market events and ranks stocks for the next U.S. trading session "
    "using event + momentum scoring."
)

default_newsapi_key = os.getenv("NEWSAPI_KEY", "")
default_openai_key = os.getenv("OPENAI_API_KEY", "")
default_openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

with st.sidebar:
    st.header("Settings")
    lookback_hours = st.slider("Event lookback window (hours)", 6, 72, DEFAULT_LOOKBACK_HOURS, step=6)
    num_picks = st.slider("Number of stock picks", 3, 15, DEFAULT_NUM_PICKS)

    newsapi_key = st.text_input("NewsAPI key (optional)", value=default_newsapi_key, type="password")

    default_mode = "Auto (GPT if key provided)" if default_openai_key else "Heuristic only"
    sentiment_mode = st.selectbox(
        "Sentiment mode",
        SENTIMENT_MODE_OPTIONS,
        index=SENTIMENT_MODE_OPTIONS.index(default_mode),
        help="Auto: use GPT when key is present. GPT only: force GPT attempt. Heuristic only: disable GPT.",
    )

    openai_api_key = st.text_input("OpenAI API key", value=default_openai_key, type="password")
    openai_model = st.text_input("OpenAI model", value=default_openai_model)

    st.markdown("### Usage")
    st.write("Run this after market close or before next market open.")
    st.caption(
        "RSS feeds are rolling latest-N streams. Increasing lookback (for example 24h to 72h) may not increase "
        "event count if sources do not expose older items."
    )

run_scan = st.button("Run Next-Day Scan", type="primary")

if run_scan:
    resolved_openai_key = (openai_api_key or "").strip()
    resolved_openai_model = (openai_model or "").strip() or "gpt-4o-mini"

    if sentiment_mode == "Heuristic only":
        effective_use_gpt = False
    elif sentiment_mode == "GPT only (require key)":
        effective_use_gpt = True
    else:
        effective_use_gpt = bool(resolved_openai_key)

    if sentiment_mode == "GPT only (require key)" and not resolved_openai_key:
        st.warning("GPT-only mode selected, but no OpenAI key provided. This run will fall back to heuristic.")

    with st.spinner("Collecting events and building recommendations..."):
        result = build_next_day_recommendations(
            lookback_hours=lookback_hours,
            num_picks=num_picks,
            newsapi_key=newsapi_key or None,
            use_gpt_sentiment=effective_use_gpt,
            openai_api_key=resolved_openai_key or None,
            openai_model=resolved_openai_model,
        )

    generated_at = result["generated_at_utc"]
    next_day = result["next_trading_day"]
    events = result["events"]
    signals_df: pd.DataFrame = result["signals_df"]
    rec_df: pd.DataFrame = result["recommendations_df"]

    sentiment_backend = result.get("sentiment_backend", "heuristic")
    sentiment_backend_reason = result.get("sentiment_backend_reason", "")
    used_model = result.get("openai_model", resolved_openai_model)
    sentiment_stats = result.get("sentiment_stats", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Hot events scanned", len(events))
    c2.metric("Stocks recommended", len(rec_df))
    c3.metric("Target trading day", str(next_day))

    st.caption(f"Generated at (UTC): {generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
    st.caption(
        "Note: Event count depends on source feed retention. RSS providers often expose only recent items, "
        "so larger lookback windows may return the same count."
    )

    st.subheader("Sentiment Run Info")
    key_source = _detect_key_source(resolved_openai_key, default_openai_key.strip())

    info_cols = st.columns(2)
    info_cols[0].metric("Sentiment mode selected", sentiment_mode)
    info_cols[1].metric("OpenAI key source", key_source)

    st.markdown(f"**Actual backend used:** `{sentiment_backend}`")
    if sentiment_backend_reason:
        st.caption(sentiment_backend_reason)

    gpt_calls = int(sentiment_stats.get("gpt_api_calls", 0) or 0)
    gpt_success = int(sentiment_stats.get("gpt_success_events", 0) or 0)
    gpt_failed = int(sentiment_stats.get("gpt_failed_events", 0) or 0)
    gpt_cache_hits = int(sentiment_stats.get("gpt_cache_hits", 0) or 0)
    fallback_events = int(sentiment_stats.get("heuristic_fallback_events", 0) or 0)
    prompt_tokens = int(sentiment_stats.get("prompt_tokens", 0) or 0)
    completion_tokens = int(sentiment_stats.get("completion_tokens", 0) or 0)
    total_tokens = int(sentiment_stats.get("total_tokens", 0) or 0)

    if effective_use_gpt:
        st.caption(f"GPT model setting: {used_model}")
        usage_cols = st.columns(4)
        usage_cols[0].metric("GPT-scored events", gpt_success)
        usage_cols[1].metric("GPT API calls", gpt_calls)
        usage_cols[2].metric("Cache hits", gpt_cache_hits)
        usage_cols[3].metric("Fallback events", fallback_events + gpt_failed)

        token_cols = st.columns(3)
        token_cols[0].metric("Prompt tokens", prompt_tokens)
        token_cols[1].metric("Completion tokens", completion_tokens)
        token_cols[2].metric("Total tokens", total_tokens)

        estimated_cost = _estimate_cost_usd(sentiment_stats, used_model)
        pricing = _resolve_pricing(used_model)
        if estimated_cost is None:
            st.info(
                "Run-level cost estimate unavailable for this model name. Token usage is shown above; "
                "check current model pricing on the OpenAI API pricing page."
            )
        else:
            st.info(
                f"Estimated OpenAI API cost for this run: ${estimated_cost:.6f} USD (estimate based on token usage; "
                "final billed amount may differ slightly)."
            )
        if pricing is not None:
            st.caption(
                f"Cost-rate assumption for estimate: input ${pricing['input']}/1M tokens, "
                f"output ${pricing['output']}/1M tokens (pricing snapshot date: 2026-03-06)."
            )

    if rec_df.empty:
        st.warning("No recommendations were produced. Try expanding the lookback window.")
    else:
        display = rec_df.copy()
        pct_cols = ["ret_1d", "ret_5d", "ret_20d", "confidence", "portfolio_weight"]
        for col in pct_cols:
            if col in display:
                display[col] = (display[col] * 100).map(lambda x: f"{x:.2f}%")

        if "vol_ratio" in display:
            display["vol_ratio"] = display["vol_ratio"].map(lambda x: f"{x:.2f}x")
        if "last_close" in display:
            display["last_close"] = display["last_close"].map(lambda x: f"${x:.2f}")
        if "event_score" in display:
            display["event_score"] = display["event_score"].map(lambda x: f"{x:+.3f}")
        if "raw_event_sentiment" in display:
            display["raw_event_sentiment"] = display["raw_event_sentiment"].map(lambda x: f"{x:+.3f}")

        st.subheader("Top Picks and Suggested Next-Day Portfolio Weights")
        st.dataframe(
            display[
                [
                    "rank",
                    "ticker",
                    "portfolio_weight",
                    "score",
                    "raw_event_sentiment",
                    "event_score",
                    "confidence",
                    "event_count",
                    "themes",
                    "last_close",
                    "ret_1d",
                    "ret_5d",
                    "vol_ratio",
                    "catalysts",
                ]
            ].rename(
                columns={
                    "raw_event_sentiment": "avg_sentiment",
                    "event_score": "weighted_event",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Summary Generation Info")
        use_gpt_summary = bool(resolved_openai_key)
        summary_results = build_recommendation_summaries(
            rec_df=rec_df,
            signals_df=signals_df,
            target_day=next_day,
            use_gpt_summary=use_gpt_summary,
            openai_api_key=resolved_openai_key or None,
            openai_model=resolved_openai_model,
        )

        summary_gpt = sum(1 for value in summary_results.values() if value.get("backend") == "gpt")
        summary_fallback = sum(1 for value in summary_results.values() if value.get("backend") == "heuristic_fallback")

        summary_cols = st.columns(3)
        summary_cols[0].metric("Summary mode", "GPT" if use_gpt_summary else "Heuristic")
        summary_cols[1].metric("GPT summaries", summary_gpt)
        summary_cols[2].metric("Fallback summaries", summary_fallback)

        if not use_gpt_summary:
            st.caption("No OpenAI API key detected, so summaries used heuristic mode.")
        elif summary_fallback > 0:
            first_error = ""
            for value in summary_results.values():
                if value.get("error"):
                    first_error = value["error"]
                    break
            if first_error:
                st.caption(f"Some summaries fell back to heuristic. Latest error: {first_error}")

        st.subheader("Why Each Stock Is Recommended")
        for _, row in rec_df.sort_values("rank").iterrows():
            ticker = str(row.get("ticker", ""))
            rank = int(row.get("rank", 0))
            payload = summary_results.get(ticker, {})
            backend = payload.get("backend", "unknown")
            text = payload.get("text", "Summary unavailable for this ticker.")

            with st.expander(f"#{rank} {ticker}", expanded=False):
                st.caption(f"Summary backend: {backend}")
                st.write(text)

    with st.expander("View Hot Events"):
        if not events:
            st.info("No events found in selected lookback window.")
        else:
            events_df = pd.DataFrame(events)
            events_df["published_utc"] = events_df["published_utc"].astype(str)
            st.dataframe(
                events_df[["published_utc", "source", "title", "url"]],
                use_container_width=True,
                hide_index=True,
            )
else:
    st.info("Click `Run Next-Day Scan` to generate recommendations.")

st.divider()
_render_methodology()
