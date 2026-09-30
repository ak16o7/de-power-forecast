"""Open-Meteo: live forecasts, archived single runs and previous-runs vintages.

Free tier (non-commercial): 600 calls/min, 5,000/h, 10,000/day, 300,000/month.
A request counts as several calls when it asks for more than 10 variables or
more than 14 days, and every location counts separately. `weight()` mirrors that
so the jobs can stay under their own, lower caps.
"""
from __future__ import annotations

import logging
import re
import time
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from .config import USER_AGENT, UTC, Point, WeatherModel

LOG = logging.getLogger(__name__)

FORECAST = "https://api.open-meteo.com/v1/forecast"
SINGLE_RUNS = "https://single-runs-api.open-meteo.com/v1/forecast"
PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
META = "https://api.open-meteo.com/data/{}/static/meta.json"

CELL = {"onshore": "land", "offshore": "sea"}


class RateLimited(Exception):
    """429 from Open-Meteo. `scope` is minute, hour, day or month."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        low = reason.lower()
        self.scope = next((s for s in ("minute", "hour", "daily", "month") if s in low), "unknown")
        if self.scope == "daily":
            self.scope = "day"


class BadRequest(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

    @property
    def bad_parameter(self) -> bool:
        """A request we built wrong (not a missing run): stop the queue instead of skipping runs."""
        low = self.reason.lower()
        return "parameter" in low or "must not be set" in low or "invalid" in low

    @property
    def bad_variable(self) -> str | None:
        m = re.search(r"invalid String value ([a-z0-9_]+)", self.reason)
        return m.group(1) if m else None


class BudgetExhausted(Exception):
    pass


def weight(n_locations: int, n_variables: int, hours: float) -> float:
    return n_locations * max(1.0, n_variables / 10) * max(1.0, hours / 24 / 14)


class Budget:
    """Fractional call accounting for one job run, plus a per-minute pacer."""

    def __init__(self, run_cap: float, day_left: float, minute_cap: float) -> None:
        self.run_cap = run_cap
        self.day_left = day_left
        self.minute_cap = minute_cap
        self.used = 0.0
        self._window: deque[tuple[float, float]] = deque()

    def left(self) -> float:
        return max(0.0, min(self.run_cap - self.used, self.day_left - self.used))

    def spend(self, w: float) -> None:
        if w > self.left():
            raise BudgetExhausted(f"need {w:.1f}, left {self.left():.1f}")
        while True:
            now = time.monotonic()
            while self._window and now - self._window[0][0] > 60:
                self._window.popleft()
            if sum(x for _, x in self._window) + w <= self.minute_cap:
                break
            time.sleep(max(1.0, 60 - (now - self._window[0][0]) + 0.5))
        self._window.append((time.monotonic(), w))
        self.used += w


class Client:
    def __init__(self, budget: Budget | None = None, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.budget = budget
        self.calls = 0
        self.weight_used = 0.0

    # ------------------------------------------------------------------ raw
    def _get(self, url: str, params: dict, w: float):
        if self.budget:
            self.budget.spend(w)
        self.calls += 1
        self.weight_used += w
        last = ""
        for attempt in range(4):
            try:
                r = self.session.get(url, params=params, timeout=120)
            except requests.RequestException as exc:
                last = type(exc).__name__
            else:
                if r.status_code == 200:
                    try:
                        return r.json()
                    except ValueError:   # cut-off body (seen once on a 1.5 MB answer): ask again
                        last = f"truncated JSON ({len(r.content)} bytes)"
                        time.sleep(5 * 2 ** attempt)
                        continue
                reason = _reason(r)
                if r.status_code == 429:
                    err = RateLimited(reason)
                    if err.scope != "minute" or attempt == 3:
                        raise err
                    time.sleep(65)
                    continue
                if r.status_code == 400:
                    raise BadRequest(reason)
                last = f"HTTP {r.status_code}: {reason[:120]}"
            time.sleep(5 * 2 ** attempt)
        raise RuntimeError(f"Open-Meteo failed: {last}")

    def meta(self, model: WeatherModel) -> dict | None:
        """Run bookkeeping of a model (static file, does not count as an API call)."""
        self.meta_error = None
        for attempt in range(3):
            try:
                r = self.session.get(META.format(model.meta_dir), timeout=60)
                if r.status_code == 200:
                    return r.json()
                self.meta_error = f"HTTP {r.status_code}"
                if r.status_code == 404:
                    return None
            except (requests.RequestException, ValueError) as exc:
                self.meta_error = type(exc).__name__
            time.sleep(3 * (attempt + 1))
        return None

    # ------------------------------------------------------------------ products
    def forecast(self, model: WeatherModel, points, start: datetime, hours: int,
                 variables: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Latest run of `model` for [start, start + hours)."""
        return self._points(FORECAST, model, points, variables or model.variables, start, hours, {})

    def single_run(self, model: WeatherModel, run: datetime, points, hours: int | None = None,
                   variables: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Archived run initialised at `run` (UTC), `hours` steps from the run start.

        The Single Runs API refuses start_hour/end_hour; forecast_hours counts from the run.
        """
        hours = hours or model.horizon_h
        extra = {"run": run.strftime("%Y-%m-%dT%H:%M"), "forecast_hours": hours}
        return self._points(SINGLE_RUNS, model, points, variables or model.variables, run, hours, extra,
                            window=False)

    def previous_runs(self, model: WeatherModel, points, start: datetime, end: datetime,
                      days: tuple[int, ...], variables: tuple[str, ...] | None = None) -> pd.DataFrame:
        """Values predicted >= N*24 h before their valid time, for valid times [start, end)."""
        base = variables or model.variables
        hourly = [f"{v}_previous_day{d}" for d in days for v in base]
        hours = int((end - start) / timedelta(hours=1))
        wide = self._points(PREVIOUS_RUNS, model, points, tuple(hourly), start, hours, {})
        frames = []
        for d in days:
            cols = {f"{v}_previous_day{d}": v for v in base if f"{v}_previous_day{d}" in wide}
            part = wide[["point", "valid", *cols]].rename(columns=cols)
            part.insert(2, "lead_days", np.int8(d))
            frames.append(part)
        return pd.concat(frames, ignore_index=True)

    def _points(self, url, model, points, variables, start: datetime, hours: int, extra: dict,
                window: bool = True) -> pd.DataFrame:
        start = start.astimezone(UTC)
        end = start + timedelta(hours=hours - 1)
        span = {"start_hour": start.strftime("%Y-%m-%dT%H:%M"), "end_hour": end.strftime("%Y-%m-%dT%H:%M")} \
            if window else {}
        frames = []
        groups: dict[str, list[Point]] = {}
        for p in points:
            groups.setdefault(p.kind, []).append(p)
        for kind, pts in groups.items():
            params = {
                "latitude": ",".join(f"{p.lat}" for p in pts),
                "longitude": ",".join(f"{p.lon}" for p in pts),
                "hourly": ",".join(variables),
                "models": model.name,
                "timezone": "GMT",
                "timeformat": "unixtime",
                "cell_selection": CELL.get(kind, "nearest"),
                **span,
                **extra,
            }
            data = self._get(url, params, weight(len(pts), len(variables), hours))
            frames.append(_frame(data, pts, variables))
        return pd.concat(frames, ignore_index=True)


def _reason(r: requests.Response) -> str:
    try:
        return str(r.json().get("reason", ""))[:300]
    except ValueError:
        return r.text[:300]


def _frame(data, pts: list[Point], variables) -> pd.DataFrame:
    items = data if isinstance(data, list) else [data]
    if len(items) != len(pts):
        raise RuntimeError(f"expected {len(pts)} locations, got {len(items)}")
    frames = []
    for p, item in zip(pts, items):
        h = item.get("hourly") or {}
        t = pd.to_datetime(pd.Series(h.get("time", []), dtype="int64"), unit="s", utc=True)
        df = pd.DataFrame({"point": p.name, "valid": t})
        for v in variables:
            if v in h:
                df[v] = pd.to_numeric(pd.Series(h[v], dtype="object"), errors="coerce").astype("float32")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["point"] = out["point"].astype("string")
    return out
