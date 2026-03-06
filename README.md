
# Next-Day Stock Recommender Dashboard

(My first vibe coding exercise :)

A lightweight web app that:
1. Pulls the most recent hot market events (RSS by default, optional NewsAPI).
2. Maps events to impacted stocks.
3. Scores stocks for the **next trading day** using event sentiment + momentum + volume context.
4. Displays ranked picks in an interactive dashboard.
5. Generates a short rationale for each recommended stock.

## Features
- Interactive dashboard built with `Streamlit`
- One-click scan for next-day recommendations
- Evidence-backed picks with catalyst headlines
- `Top Picks` table includes sentiment scores (`avg_sentiment`, `weighted_event`) and suggested next-day portfolio weights (`portfolio_weight`)
- GPT-generated recommendation summaries when API key is available
- Each summary includes timestamped news references (UTC)
- Per-run sentiment diagnostics: selected mode, actual backend used, backend reason, key source, token usage, estimated GPT API cost
- Methodology section in-dashboard with explicit score formulas
- Works without paid data by default (RSS + Yahoo Finance price data)
- Optional `NEWSAPI_KEY` support for broader event coverage

## Project Structure
- `app.py`: Streamlit dashboard
- `src/data_sources.py`: Event collection from RSS / NewsAPI
- `src/event_parser.py`: Ticker extraction, event theme, sentiment (heuristic + GPT + telemetry)
- `src/scoring.py`: Price feature engineering, ranking, and portfolio-weight generation
- `src/recommender.py`: End-to-end recommendation pipeline
- `src/explanations.py`: Per-stock recommendation summary generator (GPT + fallback)
- `src/config.py`: Feeds, ticker universe, scoring dictionaries

## Setup (PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If script activation is blocked, run this once in the same terminal:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Optional environment variables:
```bash
copy .env.example .env
# set NEWSAPI_KEY
# set OPENAI_API_KEY and optionally OPENAI_MODEL for GPT features
```

## Run
```bash
python -m streamlit run app.py
```

## Sentiment Mode
In the sidebar, choose one mode:
1. `Auto (GPT if key provided)`
2. `GPT only (require key)`
3. `Heuristic only`

After each run, check `Sentiment Run Info` for:
- full backend code (for example `gpt`, `gpt_with_partial_fallback`, `heuristic_due_to_gpt_errors`)
- plain-English backend reason (including latest GPT error when relevant)
- key source (`dashboard input` vs `.env`)
- token usage and estimated run cost

## Portfolio Weights
- The app generates long-only next-day portfolio weights from the selected picks.
- Weights are normalized to sum to 100%.
- Formula (also shown in dashboard Methodology):
  - `portfolio_signal = ((score - min(score)) + 0.05) * (0.5 + confidence)`
  - `portfolio_weight = portfolio_signal / sum(portfolio_signal)`

## Summary Generation
- If an OpenAI API key is available, summaries are generated with GPT.
- If GPT summary generation fails, the app falls back to heuristic summary and shows the fallback count/error.
- All referenced news items are shown with UTC timestamps.

## Stop the App
In the same terminal where Streamlit is running, press:
```bash
Ctrl + C
```

## Recommended Usage Window
Run it:
- After U.S. market close (around 4:00 PM ET), or
- Before the next market open.

The app automatically outputs recommendations for the next U.S. trading day (weekend-aware).

## Important Notes
- This is a rule-based MVP, not financial advice.
- GPT sentiment/summaries can improve context interpretation, but add API latency/cost.
- Cost shown is an estimate from token usage and configured model pricing assumptions.
- RSS feeds are rolling latest-N streams; increasing lookback (for example 24h to 72h) may not increase event count if providers do not expose older items.
- U.S. market holidays are not yet modeled (weekends only).
- For production use, add robust data validation, caching, testing, and risk controls.

