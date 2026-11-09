"""Turn raw signals (weather, events, holidays, search trends) into the
`CityResult` objects the frontend renders on the heatmap.

Shape contract: frontend/src/lib/citypulse/types.ts -> CityResult
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from backend import config
from backend.services.compliance_rules import check_copy
from ml.agent.decide import get_local_context
from ml.signals import city_trends
from ml.signals.events import get_real_events
from ml.signals.open_meteo import WEATHER_DESCRIPTIONS, _get_json

ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)

THRESHOLDS = {"generate": 60, "light_touch": 30}  # keep in sync with frontend config.ts
ACTION_COST = {
    "GENERATE": (0.14, 45),
    "LIGHT_TOUCH": (0.03, 20),
    "SKIP": (0.0, 0),
}
WINDOW_DAYS = 3          # weather
HOLIDAY_WINDOW_DAYS = 5  # holidays: a little lead time for greetings
EVENT_WINDOW_DAYS = 21   # events: ads need lead time (big markets are announced weeks ahead)

# ---------------------------------------------------------------- caches
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()
_key_locks: dict[str, threading.Lock] = {}


def _cached(key: str, ttl: float, producer, fail_ttl: float = 300):
    """Return producer() cached for ttl seconds. Errors are swallowed, returned as None
    and cached for fail_ttl so a down API is not hammered. One producer per key at a time."""

    def fresh():
        hit = _cache.get(key)
        if not hit:
            return None
        ts, value = hit
        return hit if time.time() - ts < (ttl if value is not None else fail_ttl) else None

    with _cache_lock:
        lock = _key_locks.setdefault(key, threading.Lock())
    with lock:
        with _cache_lock:
            hit = fresh()
        if hit:
            return hit[1]
        old = _cache.get(key)
        ts = time.time()
        try:
            value = producer()
        except Exception as error:
            log.warning("%s failed: %s", key, str(error)[:200])
            value = None
            if old and old[1] is not None:  # keep serving the last good value, retry after fail_ttl
                value = old[1]
                ts = time.time() - ttl + fail_ttl
        with _cache_lock:
            _cache[key] = (ts, value)
        return value


def load_cities() -> list[dict[str, Any]]:
    with (ROOT / "data" / "ro_cities.json").open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- weather
def _fetch_weather(city: dict[str, Any]) -> list[dict[str, Any]]:
    data = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,precipitation_sum,wind_speed_10m_max",
            "forecast_days": WINDOW_DAYS,
            "timezone": "auto",
        },
    )
    daily = data["daily"]
    out = []
    for i, day in enumerate(daily["time"]):
        code = int(daily["weather_code"][i] or 0)
        out.append(
            {
                "day": day,
                "tmax": float(daily["temperature_2m_max"][i]),
                "tmin": float(daily["temperature_2m_min"][i]),
                "rain_prob": int(daily["precipitation_probability_max"][i] or 0),
                "rain_mm": float(daily["precipitation_sum"][i] or 0),
                "wind": float(daily["wind_speed_10m_max"][i] or 0),
                "code": code,
                "desc": WEATHER_DESCRIPTIONS.get(code, "Unknown conditions"),
            }
        )
    return out


def get_weather(city: dict[str, Any]) -> list[dict[str, Any]] | None:
    return _cached(f"wx:{city['id']}", 900, lambda: _fetch_weather(city))


CLIMATE_YEARS = 3
CLIMATE_WINDOW = 8  # +/- days around today, in each past year


def _fetch_climate(city: dict[str, Any]) -> dict[str, float]:
    """This city's own normal for the time of year: mean/spread of temperature and how often it rains,
    from the last CLIMATE_YEARS years of Open-Meteo archive data around today's date."""
    end = date.today() - timedelta(days=7)  # the archive lags a few days
    start = end - timedelta(days=365 * CLIMATE_YEARS)
    data = _get_json(
        "https://archive-api.open-meteo.com/v1/archive",
        {
            "latitude": city["latitude"],
            "longitude": city["longitude"],
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "timezone": "auto",
        },
    )["daily"]
    today_doy = date.today().timetuple().tm_yday
    tmax, tmin, rain = [], [], []
    for i, day in enumerate(data["time"]):
        doy = date.fromisoformat(day).timetuple().tm_yday
        if min(abs(doy - today_doy), 365 - abs(doy - today_doy)) > CLIMATE_WINDOW:
            continue
        if data["temperature_2m_max"][i] is None or data["temperature_2m_min"][i] is None:
            continue
        tmax.append(float(data["temperature_2m_max"][i]))
        tmin.append(float(data["temperature_2m_min"][i]))
        rain.append(float(data["precipitation_sum"][i] or 0))
    if len(tmax) < 10:
        raise RuntimeError("not enough climate data")

    def mean(v: list[float]) -> float:
        return sum(v) / len(v)

    def sd(v: list[float]) -> float:
        m = mean(v)
        return (sum((x - m) ** 2 for x in v) / len(v)) ** 0.5

    rain_sorted = sorted(rain)
    return {
        "tmax_mean": mean(tmax),
        "tmax_sd": sd(tmax),
        "tmin_mean": mean(tmin),
        "tmin_sd": sd(tmin),
        "wet_rate": sum(1 for r in rain if r >= 1.0) / len(rain),
        "rain_p90": rain_sorted[int(0.9 * (len(rain_sorted) - 1))],
    }


def get_climate(city: dict[str, Any]) -> dict[str, float] | None:
    return _cached(f"clim:{city['id']}", 24 * 3600, lambda: _fetch_climate(city))


def warm_climate() -> None:
    for city in load_cities():
        get_climate(city)


def weather_signal(days: list[dict[str, Any]] | None, climate: dict[str, float] | None = None) -> tuple[dict[str, Any] | None, str]:
    """-> (signal, motif hint). motif hint is 'storm' | 'heat' | 'plain'.

    With a climate baseline the strength depends on how unusual the weather is FOR THIS CITY at this
    time of year (normal weather scores 0). Without one it falls back to fixed absolute rules."""
    if not days:
        return None, "plain"
    worst_rain = max(days, key=lambda d: (d["rain_prob"], d["rain_mm"]))
    hottest = max(days, key=lambda d: d["tmax"])
    coldest = min(days, key=lambda d: d["tmin"])
    stormy = worst_rain["code"] >= 95 or worst_rain["rain_mm"] >= 10 or worst_rain["wind"] >= 60  # severe anywhere

    if climate:
        hot_dev = hottest["tmax"] - climate["tmax_mean"]
        cold_dev = climate["tmin_mean"] - coldest["tmin"]
        hot_z = hot_dev / max(2.5, climate["tmax_sd"])
        cold_z = cold_dev / max(2.5, climate["tmin_sd"])
        wet_excess = worst_rain["rain_prob"] / 100 - climate["wet_rate"]
        rain_unusual = worst_rain["rain_prob"] >= 50 and (
            wet_excess >= 0.3 or worst_rain["rain_mm"] >= max(5.0, climate["rain_p90"])
        )
        if stormy or rain_unusual:
            strength = 45 if stormy else min(40, round(16 + wet_excess * 55))
            label = f"{worst_rain['desc']}, {worst_rain['rain_prob']}% rain (usually {round(climate['wet_rate'] * 100)}% of days)"
            return {"type": "weather", "label": label, "source": "live", "strength": strength}, "storm"
        if hot_z >= 1.0 and hottest["tmax"] >= 22:
            strength = min(45, round(14 + (hot_z - 1.0) * 14))
            label = f"Heat, {round(hottest['tmax'])}°C ({hot_dev:+.0f}° vs this city's normal)"
            return {"type": "weather", "label": label, "source": "live", "strength": strength}, "heat"
        if cold_z >= 1.5:
            strength = min(35, round(12 + (cold_z - 1.5) * 10))
            label = f"Colder than usual, {round(coldest['tmin'])}°C ({-cold_dev:+.0f}° vs this city's normal)"
            return {"type": "weather", "label": label, "source": "live", "strength": strength}, "plain"
        today = days[0]
        return (
            {"type": "weather", "label": f"{today['desc']}, {round(today['tmax'])}°C, normal for the season", "source": "live", "strength": 0},
            "plain",
        )

    # --- no baseline available: fixed rules ---
    if stormy or worst_rain["rain_prob"] >= 70:
        strength = 45 if stormy else 32
        label = f"{worst_rain['desc']}, {worst_rain['rain_prob']}% rain"
        return {"type": "weather", "label": label, "source": "live", "strength": strength}, "storm"
    if hottest["tmax"] >= 30:
        strength = min(45, 25 + (hottest["tmax"] - 30) * 4)
        return (
            {"type": "weather", "label": f"Heat, {round(hottest['tmax'])}°C", "source": "live", "strength": round(strength)},
            "heat",
        )
    if coldest["tmin"] <= 0:
        return (
            {"type": "weather", "label": f"Freezing, {round(coldest['tmin'])}°C", "source": "live", "strength": 22},
            "plain",
        )
    today = days[0]
    return (
        {"type": "weather", "label": f"{today['desc']}, {round(today['tmax'])}°C", "source": "live", "strength": 5},
        "plain",
    )


# ---------------------------------------------------------------- trends
def _norm(text: str) -> str:
    folded = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return "".join(c for c in folded.casefold() if c.isalnum())


def _with_timeout(fn, seconds: float):
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=seconds)
    finally:
        pool.shutdown(wait=False)


TREND_FRESH_S = 6 * 3600          # reuse a fetched result this long
TREND_STALE_OK_S = 7 * 24 * 3600  # when Google blocks us, older real data still beats nothing
TREND_BLOCK_S = 15 * 60           # pause all Google requests this long after being blocked
TREND_CACHE_FILE = config.STORAGE_DIR / "trends_cache.json"
_disk_lock = threading.Lock()
_trends_blocked_until = 0.0


def _disk_get(key: str) -> tuple[float, Any] | None:
    try:
        with _disk_lock:
            entry = json.loads(TREND_CACHE_FILE.read_text(encoding="utf-8")).get(key)
    except (OSError, ValueError):
        return None
    return (float(entry["ts"]), entry["value"]) if entry else None


def _disk_put(key: str, value: Any) -> None:
    try:
        with _disk_lock:
            try:
                data = json.loads(TREND_CACHE_FILE.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            data[key] = {"ts": time.time(), "value": value}
            tmp = TREND_CACHE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            tmp.replace(TREND_CACHE_FILE)
    except OSError:
        log.warning("could not write %s", TREND_CACHE_FILE)


def _fetch_trend_resilient(city: dict[str, Any], category: str) -> dict[str, Any]:
    """Google Trends for a city: fresh disk copy -> live fetch -> stale disk copy when Google blocks us."""
    global _trends_blocked_until
    key = f"ctrend:{city['id']}:{category}"
    stored = _disk_get(key)
    if stored and time.time() - stored[0] < TREND_FRESH_S:
        return stored[1]
    try:
        if time.time() < _trends_blocked_until:
            raise RuntimeError("Google Trends is blocking requests, backing off")
        value = _with_timeout(lambda: city_trends.fetch_city_trend(city["id"], category), 30)
    except Exception as exc:
        text = str(exc)
        if "sorry" in text or "429" in text or "Too Many" in text:
            _trends_blocked_until = time.time() + TREND_BLOCK_S
        log.warning("Google Trends failed for %s (%s): %s", city["id"], category, text[:140])
        if stored and time.time() - stored[0] < TREND_STALE_OK_S:
            return stored[1]  # last real data we have
        raise
    _disk_put(key, value)
    return value


def _live_city_trend(city: dict[str, Any], category: str) -> dict[str, Any] | None:
    """Google Trends for this city's county (cached; None when Google gives us nothing)."""
    if os.getenv("CITYPULSE_LIVE_TRENDS", "1") == "0" or city["id"] not in city_trends.GEO:
        return None
    return _cached(
        f"ctrend:{city['id']}:{category}",
        TREND_FRESH_S,
        lambda: _fetch_trend_resilient(city, category),
        fail_ttl=600,
    )


def warm_trends(category: str = "Cafe") -> None:
    """Fill the trend cache in the background (called at startup) so the first request is fast."""
    for city in load_cities():
        _live_city_trend(city, category)


def _short(query: str, limit: int = 30) -> str:
    return query if len(query) <= limit else query[: limit - 1].rstrip() + "…"


def trend_signals(city: dict[str, Any], category: str) -> list[dict[str, Any]]:
    """Search signals for a city: how the category's searches moved, plus its popular local query."""
    data = _live_city_trend(city, category)
    if not data:
        return []  # no real search data right now: show no search signal rather than an invented one
    momentum = float(data["momentum"])
    pct = round(momentum * 100)
    if momentum >= 0.1:
        label = f'Searches for "{data["keyword"]}" up {pct}%'
    elif momentum <= -0.1:
        label = f'Searches for "{data["keyword"]}" down {abs(pct)}%'
    else:
        label = f'Searches for "{data["keyword"]}" steady'
    # Only the change against this city's own recent history counts, not how big the city's search volume is.
    strength = 3 + momentum * 45
    signals = [{"type": "trend", "label": label, "source": "live", "strength": round(max(0, min(28, strength)))}]
    if data["rising"]:
        signals.append({"type": "trend", "label": f'Rising search: "{_short(data["rising"][0][0])}"', "source": "live", "strength": 6})
    elif data["top"]:
        signals.append({"type": "trend", "label": f'Top search: "{_short(data["top"][0][0])}"', "source": "live", "strength": 2})
    return signals


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first of events whose names contain each other ("X" and "X Opening")."""
    kept: list[dict[str, Any]] = []
    for ev in events:
        name = _norm(str(ev.get("name", "")))
        if any(name and (name in _norm(str(k.get("name", ""))) or _norm(str(k.get("name", ""))) in name) for k in kept):
            continue
        kept.append(ev)
    return kept


# ---------------------------------------------------------------- copy
COPY = {
    "storm": ("Afara ploua. Noi venim la tine.", "Comanda acum si livram oriunde in oras, uscat si la timp."),
    "heat": ("Cald afara? Ceva racoros, acum.", "Intra la noi si fa o pauza la racoare."),
    "festival": ("Weekend plin in oras? Incepe cu noi.", "Treci pe la noi inainte sa te pierzi in forfota orasului."),
    "greeting": ("O zi speciala. Ne bucuram sa te vedem.", "Vino sa sarbatorim impreuna, cum stim noi mai bine."),
    "plain": (None, None),  # master copy is kept
}

MASTER = {
    "Cafe": ("Cafeaua ta, in fiecare oras.", "Boabe proaspat prajite, servite cu zambet."),
    "Pharmacy": ("Grija pentru tine, aproape de casa.", "Sfaturi de la farmacisti, in fiecare cartier."),
    "Florist": ("Flori proaspete pentru fiecare zi.", "Buchete facute de mana, livrate azi."),
    "Gym": ("Mai puternic in fiecare zi.", "Antrenamente pentru orice program."),
    "Supplements": ("Energie pentru ritmul tau.", "Formule simple, ingrediente clare."),
}


def _when(iso: str) -> str:
    d = date.fromisoformat(iso)
    if d <= date.today():
        return "now"
    return f"{d.day} {d.strftime('%b')}"


def decide(score: float) -> str:
    if score >= THRESHOLDS["generate"]:
        return "GENERATE"
    if score >= THRESHOLDS["light_touch"]:
        return "LIGHT_TOUCH"
    return "SKIP"


_event_stats: dict[str, Any] = {}


def _event_factor(city_name: str, relevance: float) -> float:
    """How notable an event is compared with this city's other events (z-score vs the city's own pool,
    shrunk toward the global pool because the city may only have a few). 1.0 = typical."""
    if not _event_stats:
        try:
            with (ROOT / "data" / "ro_events.json").open(encoding="utf-8") as f:
                items = json.load(f).get("events", [])
        except (OSError, ValueError):
            items = []
        by_city: dict[str, list[float]] = {}
        for it in items:
            by_city.setdefault(_norm(str(it.get("city", ""))), []).append(float(it.get("relevance", 0.5)))
        allv = [v for vs in by_city.values() for v in vs] or [0.6]
        _event_stats.update(by_city=by_city, g_mean=sum(allv) / len(allv))
    own = _event_stats["by_city"].get(_norm(city_name), [])
    k = 3  # pseudo-events from the global pool
    n = len(own)
    mean = (sum(own) + k * _event_stats["g_mean"]) / (n + k)
    z = (relevance - mean) / 0.2
    return max(0.5, min(1.3, 1 + 0.35 * z))


def build_city_result(city: dict[str, Any], category: str) -> dict[str, Any]:
    category = category if category in MASTER else "Cafe"
    weather = get_weather(city)
    wx_signal, wx_motif = weather_signal(weather, get_climate(city))

    try:
        ctx = get_local_context(city["name"], HOLIDAY_WINDOW_DAYS)
    except Exception:
        ctx = {"events": [], "holidays": [], "namedays": []}

    try:
        events = _dedupe_events(get_real_events(city["name"], EVENT_WINDOW_DAYS)["events"])
    except Exception:
        events = []

    signals: list[dict[str, Any]] = [wx_signal] if wx_signal else []
    top_event = None
    soon = (date.today() + timedelta(days=WINDOW_DAYS)).isoformat()
    for ev in events[:2]:  # already sorted by relevance
        top_event = top_event or ev
        # Already running or starting within the weather window counts fully; later events count less.
        if ev["start_date"] <= soon:
            proximity = 1.0
        elif ev["start_date"] <= (date.today() + timedelta(days=10)).isoformat():
            proximity = 0.8
        else:
            proximity = 0.55
        signals.append(
            {
                "type": "event",
                "label": f"{ev.get('name', 'Local event')} · {_when(ev['start_date'])}",
                "source": ev.get("source", "curated"),
                "strength": round(min(50.0, (15 + float(ev.get("relevance", 0.5)) * 35) * proximity * _event_factor(city["name"], float(ev.get("relevance", 0.5))))),
            }
        )
    top_holiday = None
    for hol in ctx.get("holidays", [])[:1]:
        top_holiday = hol
        signals.append(
            {"type": "holiday", "label": str(hol.get("name", "Public holiday")), "source": "curated", "strength": 30}
        )
    signals.extend(trend_signals(city, category))

    score = max(0, min(100, round(sum(s["strength"] for s in signals))))
    action = decide(score)

    if wx_motif == "storm":
        motif, reason = "storm", f"{wx_signal['label']} - people stay in, so the ad switches to delivery."
    elif top_event:
        motif, reason = "festival", f"{top_event.get('name', 'A local event')} fills the city, so a festive visual and copy are worth making."
    elif top_holiday:
        motif, reason = "greeting", f"{top_holiday.get('name', 'A holiday')} is coming up, so a greeting tone fits."
    elif wx_motif == "heat":
        motif, reason = "heat", f"{wx_signal['label']} - the visual and copy turn cool."
    else:
        motif, reason = "plain", "Nothing unusual is happening, so the master ad already fits."
    if action == "SKIP":
        reason = "Nothing strong enough is happening, so the master ad already fits."
        motif = "plain"

    master_headline, master_body = MASTER[category]
    local_headline, local_body = COPY[motif]
    headline = local_headline or master_headline
    body = local_body or master_body
    cost, minutes = ACTION_COST[action]

    compliance = check_copy(headline, body, category=category)
    if compliance["status"] == "fixed":  # use the auto-softened copy; the UI only needs status + note
        headline = compliance.pop("headline")
        body = compliance.pop("body")

    return {
        "city": city["name"],
        "lat": city["latitude"],
        "lon": city["longitude"],
        "score": score,
        "action": action,
        "signals": signals,
        "reason": reason,
        "headline": headline,
        "body": body,
        "motif": motif,
        "compliance": compliance,
        "category": category,
        "costUsd": cost,
        "minutesSaved": minutes,
    }


def city_list() -> list[dict[str, Any]]:
    """Every supported city, in the shape the upload page and landing map need."""
    return [
        {"id": c["id"], "name": c["name"], "county": c["county"], "lat": c["latitude"], "lon": c["longitude"]}
        for c in load_cities()
    ]


def build_city_results(names: list[str], category: str) -> list[dict[str, Any]]:
    wanted = {n.casefold() for n in names}
    selected = [c for c in load_cities() if c["name"].casefold() in wanted]
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(lambda c: build_city_result(c, category), selected))
