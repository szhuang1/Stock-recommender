from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dateutil import parser

MARKET_TZ = ZoneInfo("America/New_York")


def parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = parser.parse(value)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
        return parsed.astimezone(ZoneInfo("UTC"))
    except Exception:
        return None


def utc_now() -> datetime:
    return datetime.now(tz=ZoneInfo("UTC"))


def get_next_trading_day(reference_utc: datetime | None = None) -> datetime.date:
    ref = reference_utc.astimezone(MARKET_TZ) if reference_utc else datetime.now(MARKET_TZ)
    day = ref.date()

    # If after 4:00 PM ET, move to next day before weekend adjustment.
    if ref.hour >= 16:
        day += timedelta(days=1)

    while day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        day += timedelta(days=1)

    return day


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
