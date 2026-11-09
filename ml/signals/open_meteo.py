"""Fetch the next few days of weather from the public Open-Meteo API."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import date
import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


WEATHER_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow fall",
    73: "Moderate snow fall",
    75: "Heavy snow fall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@dataclass(frozen=True)
class WeatherDay:
    """Daily weather values returned by Open-Meteo."""

    day: date
    temperature_max_c: float
    temperature_min_c: float
    precipitation_probability_percent: int
    precipitation_mm: float
    weather_code: int
    description: str
    wind_max_kmh: float


@dataclass(frozen=True)
class WeatherForecast:
    """Forecast and location metadata for a city."""

    source: str
    city: str
    country: str
    latitude: float
    longitude: float
    timezone: str
    forecast: list[WeatherDay]


def _get_json(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET + parse JSON. Retries timeouts, HTTP 429 and 5xx (shared hosting IPs get rate-limited)."""
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"User-Agent": "CityPulse/0.1", "Accept": "application/json"},
    )
    last: Exception | None = None
    for attempt in range(3):
        if attempt:
            time.sleep(1.5 * attempt)
        try:
            with urlopen(request, timeout=15) as response:
                return json.load(response)
        except HTTPError as error:
            last = RuntimeError(f"Open-Meteo request failed with HTTP {error.code} ({url.split('/')[2]})")
            if error.code not in (429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            last = RuntimeError(f"Could not reach Open-Meteo: {reason}")
        except json.JSONDecodeError:
            last = RuntimeError("Open-Meteo returned invalid JSON")
            break
    raise last or RuntimeError("Open-Meteo request failed")


def geocode_city(city: str, country_code: str = "RO") -> dict[str, Any]:
    """Resolve a city name to coordinates using Open-Meteo geocoding."""
    if not city.strip():
        raise ValueError("city must not be empty")
    response = _get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": city.strip(), "count": 10, "language": "en", "format": "json"},
    )
    results = response.get("results", [])
    matching_results = [
        result for result in results if result.get("country_code") == country_code
    ]
    if not matching_results:
        raise ValueError(f"City {city!r} was not found in {country_code}")
    return matching_results[0]


def fetch_weather(city: str, days: int = 5, country_code: str = "RO") -> WeatherForecast:
    """Fetch the next ``days`` calendar days for a Romanian city."""
    if not 1 <= days <= 16:
        raise ValueError("days must be between 1 and 16")

    location = geocode_city(city, country_code=country_code)
    response = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "precipitation_sum",
                    "wind_speed_10m_max",
                ]
            ),
            "forecast_days": days,
            "timezone": "auto",
        },
    )
    daily = response.get("daily")
    if not isinstance(daily, dict) or "time" not in daily:
        raise RuntimeError("Open-Meteo response did not contain daily forecast data")

    forecast: list[WeatherDay] = []
    for index, day_value in enumerate(daily["time"]):
        code = int(daily["weather_code"][index])
        forecast.append(
            WeatherDay(
                day=date.fromisoformat(day_value),
                temperature_max_c=float(daily["temperature_2m_max"][index]),
                temperature_min_c=float(daily["temperature_2m_min"][index]),
                precipitation_probability_percent=int(
                    daily["precipitation_probability_max"][index]
                ),
                precipitation_mm=float(daily["precipitation_sum"][index]),
                weather_code=code,
                description=WEATHER_DESCRIPTIONS.get(code, "Unknown conditions"),
                wind_max_kmh=float(daily["wind_speed_10m_max"][index]),
            )
        )

    return WeatherForecast(
        source="Open-Meteo",
        city=location["name"],
        country=location["country"],
        latitude=float(location["latitude"]),
        longitude=float(location["longitude"]),
        timezone=str(response.get("timezone", "auto")),
        forecast=forecast,
    )


def _text_report(result: WeatherForecast) -> str:
    lines = [
        f"Source: {result.source}",
        f"Location: {result.city}, {result.country}",
        f"Coordinates: {result.latitude:.4f}, {result.longitude:.4f}",
        f"Timezone: {result.timezone}",
        "",
        "Forecast:",
    ]
    lines.extend(
        (
            f"  {day.day}: {day.description}; "
            f"{day.temperature_min_c:.1f}-{day.temperature_max_c:.1f} C; "
            f"rain probability {day.precipitation_probability_percent}%; "
            f"precipitation {day.precipitation_mm:.1f} mm"
        )
        for day in result.forecast
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    """Fetch and save a city weather report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("city", help="Romanian city, for example Bucharest or Cluj-Napoca")
    parser.add_argument("--days", type=int, default=5, help="forecast length from 1 to 16 days")
    parser.add_argument(
        "--output",
        default="weather_forecast.json",
        help="report path ending in .json or .txt",
    )
    args = parser.parse_args()

    result = fetch_weather(args.city, days=args.days)
    payload = asdict(result)
    payload["forecast"] = [asdict(day) for day in result.forecast]
    payload["forecast"] = [
        {**day, "day": day["day"].isoformat()} for day in payload["forecast"]
    ]

    if args.output.lower().endswith(".json"):
        report = json.dumps(payload, indent=2)
    elif args.output.lower().endswith(".txt"):
        report = _text_report(result)
    else:
        raise ValueError("--output must end with .json or .txt")

    with open(args.output, "w", encoding="utf-8") as output_file:
        output_file.write(report)
    print(f"Saved weather report to {args.output}")


if __name__ == "__main__":
    main()