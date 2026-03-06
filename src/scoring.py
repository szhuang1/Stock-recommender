from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
import yfinance as yf

from src.utils import clip


def _get_ticker_history(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy().dropna(how="all")

    level0 = set(data.columns.get_level_values(0))
    level1 = set(data.columns.get_level_values(1))

    if ticker in level0:
        frame = data[ticker].copy()
        return frame.dropna(how="all")

    if ticker in level1:
        frame = data.xs(ticker, axis=1, level=1).copy()
        return frame.dropna(how="all")

    return pd.DataFrame()


def fetch_market_features(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(
            columns=[
                "ticker",
                "last_close",
                "ret_1d",
                "ret_5d",
                "ret_20d",
                "vol_ratio",
                "momentum_score",
                "vol_score",
            ]
        )

    try:
        data = yf.download(
            tickers=tickers,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="ticker",
        )
    except Exception:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        hist = _get_ticker_history(data, ticker)
        if hist.empty or "Close" not in hist:
            continue

        close = hist["Close"].dropna()
        volume = hist["Volume"].dropna() if "Volume" in hist else pd.Series(dtype=float)
        if close.empty:
            continue

        ret_1d = float(close.iloc[-1] / close.iloc[-2] - 1) if len(close) >= 2 else 0.0
        ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0
        ret_20d = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0.0

        if len(volume) >= 21:
            recent_vol = float(volume.iloc[-1])
            avg_20_vol = float(volume.iloc[-21:-1].mean())
            vol_ratio = recent_vol / avg_20_vol if avg_20_vol else 1.0
        else:
            vol_ratio = 1.0

        momentum_score = clip(2.0 * ret_1d + 2.5 * ret_5d + 1.0 * ret_20d, -1.0, 1.0)
        vol_score = clip((vol_ratio - 1.0) / 1.5, -1.0, 1.0)

        rows.append(
            {
                "ticker": ticker,
                "last_close": float(close.iloc[-1]),
                "ret_1d": ret_1d,
                "ret_5d": ret_5d,
                "ret_20d": ret_20d,
                "vol_ratio": vol_ratio,
                "momentum_score": momentum_score,
                "vol_score": vol_score,
            }
        )

    return pd.DataFrame(rows)


def aggregate_recommendations(
    signals_df: pd.DataFrame,
    market_df: pd.DataFrame,
    num_picks: int,
) -> pd.DataFrame:
    event_rows: list[dict[str, Any]] = []

    if not signals_df.empty:
        by_ticker = signals_df.groupby("ticker")
        headline_map: dict[str, list[str]] = defaultdict(list)

        for _, row in signals_df.sort_values("published_utc", ascending=False).iterrows():
            ticker = row["ticker"]
            headline = row.get("headline", "")
            if headline and headline not in headline_map[ticker] and len(headline_map[ticker]) < 3:
                headline_map[ticker].append(headline)

        for ticker, group in by_ticker:
            event_rows.append(
                {
                    "ticker": ticker,
                    "event_score": float(group["weighted_event_score"].sum()),
                    "raw_event_sentiment": float(group["event_score"].mean()),
                    "event_count": int(group["headline"].count()),
                    "themes": ", ".join(sorted(set(group["theme"].astype(str).tolist()))),
                    "catalysts": " | ".join(headline_map.get(ticker, [])),
                }
            )

    event_df = pd.DataFrame(event_rows)

    if event_df.empty and market_df.empty:
        return pd.DataFrame()

    if event_df.empty:
        merged = market_df.copy()
        merged["event_score"] = 0.0
        merged["raw_event_sentiment"] = 0.0
        merged["event_count"] = 0
        merged["themes"] = "momentum"
        merged["catalysts"] = "No explicit event match; momentum-driven pick"
    else:
        merged = event_df.merge(market_df, on="ticker", how="left")

    for col in ["last_close", "ret_1d", "ret_5d", "ret_20d", "vol_ratio", "momentum_score", "vol_score"]:
        if col not in merged:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    merged["score"] = (
        0.65 * merged["event_score"]
        + 0.25 * merged["momentum_score"]
        + 0.10 * merged["vol_score"]
    )

    merged["confidence"] = (
        0.45
        + 0.10 * merged["event_count"].clip(upper=4)
        + 0.35 * merged["score"].abs().clip(upper=1.0)
    ).clip(0.0, 0.99)

    ranked = merged.sort_values("score", ascending=False).head(num_picks).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def add_portfolio_weights(recommendations_df: pd.DataFrame) -> pd.DataFrame:
    weighted = recommendations_df.copy()

    if weighted.empty:
        weighted["portfolio_weight"] = pd.Series(dtype=float)
        return weighted

    if len(weighted) == 1:
        weighted["portfolio_weight"] = 1.0
        return weighted

    score = weighted.get("score", pd.Series([0.0] * len(weighted), index=weighted.index)).astype(float)
    confidence = weighted.get("confidence", pd.Series([0.5] * len(weighted), index=weighted.index)).astype(float)

    min_score = float(score.min())
    # Shift scores so all assets have positive allocation signal; preserve relative ranking spread.
    shifted = (score - min_score) + 0.05
    conviction = 0.5 + confidence.clip(lower=0.0, upper=0.99)
    raw_signal = shifted * conviction

    if float(raw_signal.sum()) <= 0:
        weighted["portfolio_weight"] = 1.0 / len(weighted)
        return weighted

    weights = raw_signal / raw_signal.sum()
    weighted["portfolio_weight"] = weights.astype(float)
    return weighted
