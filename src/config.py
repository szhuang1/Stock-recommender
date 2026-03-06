from __future__ import annotations

RSS_FEEDS = [
    "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=%5EDJI,%5EGSPC,%5EIXIC&region=US&lang=en-US",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.investing.com/rss/news_25.rss",
]

# A focused liquid universe for a first-pass prototype.
TICKER_ALIASES = {
    "AAPL": ["apple", "iphone"],
    "MSFT": ["microsoft", "azure"],
    "NVDA": ["nvidia"],
    "AMZN": ["amazon", "aws"],
    "GOOGL": ["alphabet", "google"],
    "META": ["meta", "facebook", "instagram"],
    "TSLA": ["tesla", "elon musk"],
    "NFLX": ["netflix"],
    "AMD": ["amd", "advanced micro devices"],
    "INTC": ["intel"],
    "AVGO": ["broadcom"],
    "QCOM": ["qualcomm"],
    "ORCL": ["oracle"],
    "CRM": ["salesforce"],
    "UBER": ["uber"],
    "PYPL": ["paypal"],
    "PLTR": ["palantir"],
    "JPM": ["jpmorgan", "jp morgan"],
    "BAC": ["bank of america"],
    "WFC": ["wells fargo"],
    "GS": ["goldman sachs"],
    "MS": ["morgan stanley"],
    "V": ["visa"],
    "MA": ["mastercard"],
    "BRK-B": ["berkshire", "buffett"],
    "XOM": ["exxon", "exxonmobil"],
    "CVX": ["chevron"],
    "COP": ["conocophillips"],
    "SLB": ["schlumberger"],
    "OXY": ["occidental"],
    "LLY": ["eli lilly"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "PFE": ["pfizer"],
    "MRK": ["merck"],
    "UNH": ["unitedhealth"],
    "ABBV": ["abbvie"],
    "NVO": ["novo nordisk"],
    "PG": ["procter & gamble", "procter and gamble"],
    "KO": ["coca-cola", "coca cola"],
    "PEP": ["pepsico", "pepsi"],
    "WMT": ["walmart"],
    "COST": ["costco"],
    "HD": ["home depot"],
    "LOW": ["lowes", "lowe's"],
    "DIS": ["disney"],
    "T": ["at&t", "att"],
    "VZ": ["verizon"],
    "BA": ["boeing"],
    "GE": ["general electric", "ge aerospace"],
    "CAT": ["caterpillar"],
}

EVENT_THEMES = {
    "earnings": ["earnings", "eps", "guidance", "revenue", "forecast"],
    "m_and_a": ["acquire", "acquisition", "merger", "buyout", "deal"],
    "regulation": ["regulator", "antitrust", "ban", "lawsuit", "fine", "sec"],
    "macro": ["inflation", "cpi", "ppi", "fed", "rates", "jobless", "gdp"],
    "product": ["launch", "release", "partnership", "contract", "approval"],
    "supply_chain": ["shortage", "shipment", "inventory", "factory", "plant"],
    "geopolitics": ["tariff", "sanction", "war", "china", "taiwan", "opec"],
    "ai": ["ai", "artificial intelligence", "chip", "gpu", "data center"],
}

POSITIVE_WORDS = {
    "beat", "beats", "upgrade", "upgrades", "surge", "record", "strong", "growth",
    "profit", "profits", "bullish", "outperform", "expands", "approval", "win",
}

NEGATIVE_WORDS = {
    "miss", "misses", "downgrade", "downgrades", "drop", "weak", "loss", "bearish",
    "lawsuit", "probe", "recall", "cuts", "cut", "warning", "delay", "decline",
}

DEFAULT_LOOKBACK_HOURS = 24
DEFAULT_NUM_PICKS = 5
