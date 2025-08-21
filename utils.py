import datetime as dt
from typing import Dict
from zoneinfo import ZoneInfo

import numpy as np


def paris_now() -> dt.datetime:
    """Current time in Europe/Paris."""
    return dt.datetime.now(ZoneInfo("Europe/Paris"))


def now_iso_paris() -> str:
    """ISO datetime in Europe/Paris."""
    return paris_now().isoformat()


def infer_season_from_date(d: dt.datetime) -> str:
    """Simple season inference for EU (meteorological seasons)."""
    if d.month in (12, 1, 2):
        return "winter"
    if d.month in (3, 4, 5):
        return "spring"
    if d.month in (6, 7, 8):
        return "summer"
    return "autumn"


def infer_time_of_day(d: dt.datetime) -> str:
    """Coarse-grained moment of the day."""
    h = d.hour
    if 5 <= h < 12: return "morning"
    if 12 <= h < 17: return "afternoon"
    if 17 <= h < 22: return "evening"
    return "night"


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity with zero-safety."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_openai_headers(bearer: str) -> Dict[str, str]:
    """HTTP headers for OpenAI API."""
    return {
        "Authorization": f"Bearer {bearer}",
        "Content-Type": "application/json",
    }
