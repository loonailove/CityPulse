"""Google Trends per city: how much people in the city search for a business category,
how that changed this week, and which related searches are popular there.

Each city is queried through its county's Google Trends region code (RO-CJ for Cluj, ...),
so two cities get different numbers and different queries.
"""

from __future__ import annotations

import threading
import time
import unicodedata
from typing import Any

# Google Trends region codes (ISO 3166-2) of each city's county.
GEO = {
    "bucharest": "RO-B",
    "cluj-napoca": "RO-CJ",
    "iasi": "RO-IS",
    "constanta": "RO-CT",
    "timisoara": "RO-TM",
    "brasov": "RO-BV",
    "sibiu": "RO-SB",
    "craiova": "RO-DJ",
}

# Local-intent searches per business category (what someone types when they want to go somewhere).
KEYWORDS = {
    "Cafe": ["cafenea", "cafea"],
    "Pharmacy": ["farmacie", "raceala"],
    "Florist": ["florarie", "flori"],
    "Gym": ["sala fitness", "fitness"],
    "Supplements": ["suplimente", "proteine"],
}

# Related queries are only shown when they contain one of these (Google also returns unrelated noise).
STEMS = {
    "Cafe": ("cafe", "brunch", "ceai", "espresso", "cappuccino", "latte"),
    "Pharmacy": ("farmac", "raceal", "gripa", "vitamin", "tuse", "dureri"),
    "Florist": ("flor", "buchet", "trandafir", "lalele", "aranjament"),
    "Gym": ("fitness", "sala", "antrenament", "gym", "abonament"),
    "Supplements": ("supliment", "protein", "vitamin", "creatin", "whey"),
}

# Google throttles aggressively: at most two requests in flight, with a short pause after each.
_gate = threading.Semaphore(2)


def _norm(text: str) -> str:
    folded = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return folded.casefold()


def _rows(frame: Any) -> list[tuple[str, float]]:
    if frame is None or getattr(frame, "empty", True):
        return []
    return [(str(q), float(v)) for q, v in frame[["query", "value"]].itertuples(index=False)]


def fetch_city_trend(city_id: str, category: str) -> dict[str, Any]:
    """One Google Trends round-trip for a city. Raises on any failure (callers cache that)."""
    from pytrends.request import TrendReq

    geo = GEO[city_id]
    keywords = KEYWORDS[category]
    stems = STEMS[category]
    with _gate:
        trends = TrendReq(hl="ro-RO", tz=120, timeout=(5, 12))  # no retries=: pytrends builds urllib3 Retry(method_whitelist=...), removed in urllib3 2.x
        trends.build_payload(keywords, timeframe="today 1-m", geo=geo)
        series = trends.interest_over_time()
        related = trends.related_queries()
        time.sleep(0.6)
    if series.empty:
        raise RuntimeError(f"Google Trends returned no data for {geo}")

    values = series[keywords].mean(axis=1).tolist()
    recent = values[-7:]
    older = values[-28:-7] or recent
    recent_avg = sum(recent) / len(recent)
    older_avg = sum(older) / len(older)
    momentum = (recent_avg - older_avg) / max(older_avg, 1.0)

    def relevant(rows: list[tuple[str, float]]) -> list[tuple[str, float]]:
        return [(q, v) for q, v in rows if any(s in _norm(q) for s in stems)]

    top: list[tuple[str, float]] = []
    rising: list[tuple[str, float]] = []
    # Only the first keyword is the local-intent one ("cafenea" = a place to go); the second
    # ("cafea") mostly pulls product searches (coffee machines, supermarket offers), so it only
    # feeds the volume trend above.
    for kw in keywords[:1]:
        entry = related.get(kw) or {}
        top += relevant(_rows(entry.get("top")))
        rising += relevant(_rows(entry.get("rising")))
    top.sort(key=lambda r: -r[1])
    rising.sort(key=lambda r: -r[1])
    return {
        "geo": geo,
        "keyword": keywords[0],
        "level": recent_avg,       # 0-100, relative to the city's own busiest day this month
        "momentum": momentum,      # recent 7 days vs the 3 weeks before
        "top": top[:5],
        "rising": rising[:5],
    }
