from langchain.tools import BaseTool
import os
import re
from typing import ClassVar, Optional, List, Any, Dict
import json
import requests
from dotenv import load_dotenv
from langchain.tools.base import ToolException
from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4
from services.ingest_service import build_vector_db_from_json

load_dotenv()


class weather_tool(BaseTool):
    """
    Simple Weather Retrieval Tool using WeatherAPI.

    This tool:
      - Extracts `location`, `days`, and `query` from a natural query.
      - Calls WeatherAPI forecast endpoint for the requested period.
      - Summarizes the forecast into a short, readable answer.

    Args:
        location (str): City or place name (e.g., "Lahore").
        days (int): Number of days to fetch (1 to 10).
        query (str): Natural weather question from the user.

    Returns:
        str: A concise weather summary for each forecast day.
    """

    name: str = "weather_tool"
    description: str = (
        # "Get weather using WeatherAPI. Provide a natural question like "
        # "'weather in Lahore for next 3 days'. The tool extracts details "
        # "and returns a concise multi-day forecast."
        "Use this tool whenever the user wants to find the weather of a location. "
        "The input to this tool should always be a JSON object in string form. "
        "It must have three keys: "
        "1) 'location' → the city or country name, "
        "2) 'days' → the number of days for the forecast, "
        "3) 'query' → the exact question asked by the user. "
        "If either 'location' or 'days' is missing, set that value to 'NAN'. "
        "For example, if the user only provides a location, the input should be: "
        "{'location': 'karachi', 'days': 'NAN', 'query': 'what is the weather in karachi'}. "
        "Make sure that the JSON object is passed as a string (not as a raw JSON object)."
    )
    return_direct: bool = False

    def check_api_key(self):
        if not os.getenv("WEATHER_API"):
            raise ToolException("WEATHER_API_KEY is not set")

    def append_jsonl(self, path: str, obj: Dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")

    def _format_forecast(self, loc_name: str, days: int, data: dict) -> str:
        loc = data.get("location", {}) or {}
        resolved = (
            f"{loc.get('name', loc_name)}, {loc.get('region', '').strip()}"
        ).strip(" ,")

        fcast = ((data.get("forecast", {}) or {}).get("forecastday") or [])
        if not fcast:
            msg = f"No forecast data returned for {resolved} (days={days})."
            self.append_jsonl(
                "./data/weather/weather_formatted.jsonl",
                {
                    "request_id": data.get("request_id") or str(uuid4()),
                    "timestamp_utc": (
                        datetime.utcnow().isoformat(timespec="seconds") + "Z"
                    ),
                    "resolved_location": resolved,
                    "days_requested": days,
                    "current": None,
                    "forecast": [],
                    "message": msg,
                },
            )
            return msg

        lines: List[str] = []
        header_days = min(days, len(fcast))
        lines.append(f"Weather for {resolved} ({header_days} day(s)):")

        current = data.get("current") or {}
        if current:
            cond = ((current.get("condition") or {}).get("text") or "").strip()
            temp_c = current.get("temp_c")
            if cond or temp_c is not None:
                lines.append("Current:")
                if cond:
                    lines.append(f"- Condition: {cond}")
                if temp_c is not None:
                    lines.append(f"- Temp: {temp_c}°C")

        lines.append("Forecast:")
        compact_days: List[Dict[str, Any]] = []

        for d in fcast[:days]:
            day = d.get("day") or {}
            cond = ((day.get("condition") or {}).get("text") or "").strip()
            date = d.get("date") or ""
            maxt = day.get("maxtemp_c")
            mint = day.get("mintemp_c")
            avgt = day.get("avgtemp_c")
            precip = day.get("daily_chance_of_rain")

            lines.append(f"- {date}:")
            if cond:
                lines.append(f"  Condition: {cond}")
            if maxt is not None and mint is not None:
                lines.append(f"  High/Low: {maxt}°C / {mint}°C")
            if avgt is not None:
                lines.append(f"  Avg: {avgt}°C")
            if precip is not None:
                lines.append(f"  Rain chance: {precip}%")

            compact_days.append(
                {
                    "date": date,
                    "condition": cond or None,
                    "high_c": maxt,
                    "low_c": mint,
                    "avg_c": avgt,
                    "rain_chance_pct": precip,
                }
            )

        formatted_payload = {
            "request_id": data.get("request_id") or str(uuid4()),
            "timestamp_utc": (
                datetime.utcnow().isoformat(timespec="seconds") + "Z"
            ),
            "resolved_location": resolved,
            "days_requested": days,
            "current": (
                {
                    "condition": (
                        (current.get("condition") or {}).get("text") or None
                    ),
                    "temp_c": current.get("temp_c"),
                }
                if current
                else None
            ),
            "forecast": compact_days,
            "text": "\n".join(lines),
        }
        self.append_jsonl("./data/weather/weather_formatted.jsonl", formatted_payload)

        return "\n".join(lines)

    def _run(
        self,
        tool_input: str,
    ) -> str:
        """
        Accept either a single natural query string (ReAct style) or
        structured args (query, location, days). Returns a short summary.
        """
        self.check_api_key()
        print("before strip", tool_input)
        tool_input = tool_input.strip("'")
        print("after strip", tool_input)

        data = json.loads(tool_input)
        if data['location'].lower() == "nan":
            return "Provide location"
        if data['days'].lower() == "nan":
            return "Provide number of days"

        print("\n\n\nlocation:", data['location'])
        print("\n\n\ndays:", data['days'])
        print("\n\n\nquery:", data['query'])

        location = data['location']
        num_days = int(data['days'])
        query = str(data['query'])

        key = os.getenv("WEATHER_API")
        base_url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            "key": key,
            "q": location,
            "days": num_days,
            "aqi": "no",
            "alerts": "no",
        }

        try:
            resp = requests.get(base_url, params=params, timeout=15)
        except requests.RequestException as e:
            raise ToolException(f"Network error contacting WeatherAPI: {e}")

        if resp.status_code != 200:
            # WeatherAPI returns JSON error payloads
            try:
                err = resp.json()
            except Exception:
                err = {"error": {"message": resp.text[:200]}}
            msg = ((err.get("error") or {}).get("message") or "Unknown error")
            status = resp.status_code
            raise ToolException(
                f"WeatherAPI error ({status}): {msg}"
            )

        try:
            data = resp.json()
        except ValueError:
            raise ToolException("Invalid response from WeatherAPI (not JSON)")

        self._format_forecast(location, num_days, data)

        location_slug = str(location).strip().lower().replace(" ", "_")
        vector_db_path = f"./storage/vectordb/weather/{location_slug}_{num_days}"

        result = build_vector_db_from_json(
            pdf_paths="data/weather/weather_formatted.jsonl",
            storage_dir=vector_db_path,
            embed_model="text-embedding-3-small",
            chunk_size=1000,
            chunk_overlap=150
        )
        db_path = result.get("vector_db_path", vector_db_path)
        return (
            f"PATH={db_path}\n"
            f"QUESTION={str(query or f'weather in {location} for next {num_days} days')}"
        )
