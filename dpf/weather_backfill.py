"""Weather history without anybody's PC: an hourly GitHub Actions job that works
through fixed queues under a daily Open-Meteo budget and picks up where the last
run stopped. The queue is always "expected minus already stored", so a crashed or
skipped run loses nothing.

Queues (in config.BACKFILL_ORDER):
- previous_runs/<model>: per month, values predicted >= 24 h / >= 48 h before
  their valid time. Cheap, goes back to 2024-01. `available_at` is the safe bound
  valid - N*24 h + model delay.
- single_runs/<model>: every archived model run, same file layout as the live
  recorder (weather/runs/...). Exact vintages: available_at = run + model delay.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from . import openmeteo
from .config import (BACKFILL_ORDER, MODELS, OM_DAILY_CAP, OM_MINUTE_CAP, OM_RUN_CAP, POINTS, UTC,
                     WeatherModel)
from .record import run_path
from .store import Store
from .util import iso, months, next_month, now

LOG = logging.getLogger(__name__)
STATE = "state/weather_backfill.json"
COMMIT_EVERY = 40          # files
COMMIT_EVERY_S = 480       # seconds
EMPTY_STREAK_STOP = 8      # consecutive runs without data = start of the archive reached


def prev_path(model: str, m: datetime) -> str:
    return f"weather/previous_runs/{model}/{m:%Y}/{m:%Y-%m}.parquet"


def _vars(model: WeatherModel, q: dict) -> tuple[str, ...]:
    drop = set(q.get("drop_vars", []))
    return tuple(v for v in model.variables if v not in drop)


def _has_data(df: pd.DataFrame, variables) -> bool:
    cols = [v for v in variables if v in df]
    return bool(cols) and bool(df[cols].notna().any().any())


class Runner:
    def __init__(self, store: Store, state: dict, client: openmeteo.Client, t: datetime, max_s: float) -> None:
        self.store, self.state, self.client, self.t = store, state, client, t
        self.deadline = time.monotonic() + max_s
        self.files = 0
        self.last_commit = time.monotonic()
        self.log: list[str] = []

    def time_left(self) -> bool:
        return time.monotonic() < self.deadline

    def checkpoint(self, force: bool = False) -> None:
        due = self.store.pending >= COMMIT_EVERY or time.monotonic() - self.last_commit > COMMIT_EVERY_S
        if (force or due) and self.store.pending:
            self.state["day_used"] = round(self.state.get("day_used", 0) + self.client.weight_used - self.state.get("_counted", 0), 2)
            self.state["_counted"] = self.client.weight_used
            self.store.put_json(STATE, self.state)
            self.store.commit(f"weather backfill {iso(self.t)} ({self.files} files)")
            self.last_commit = time.monotonic()

    def fetch(self, q: dict, model: WeatherModel, call):
        """Call Open-Meteo; drop variables the endpoint does not know and retry once each."""
        for _ in range(len(model.variables)):
            try:
                return call(_vars(model, q))
            except openmeteo.BadRequest as exc:
                bad = exc.bad_variable
                base = next((v for v in model.variables if bad and bad.startswith(v)), None)
                if not base:
                    raise
                q.setdefault("drop_vars", []).append(base)
                self.log.append(f"{model.name}: dropped unsupported variable {base}")
        raise RuntimeError("no usable variables left")

    # ------------------------------------------------------------------ previous runs
    def previous_runs(self, name: str, q: dict) -> str:
        model = MODELS[name]
        last_complete = datetime(self.t.year, self.t.month, 1, tzinfo=UTC)
        if (self.t - last_complete) < timedelta(days=3):   # day2 values of the last days need 2 more days
            last_complete = datetime((last_complete - timedelta(days=1)).year,
                                     (last_complete - timedelta(days=1)).month, 1, tzinfo=UTC)
        start = datetime.fromisoformat(q["first"]) if q.get("first") else model.previous_runs_from
        present = {p.rsplit("/", 1)[-1][:7] for p in self.store.list(f"weather/previous_runs/{name}")}
        present |= set(q.get("empty", []))
        todo = [m for m in reversed(months(start, last_complete)) if f"{m:%Y-%m}" not in present]
        for m in todo:
            if not self.time_left():
                return "time"
            end = next_month(m)
            df = self.fetch(q, model, lambda v: self.client.previous_runs(model, POINTS, m, end, model.previous_days, v))
            if not _has_data(df, model.variables):
                if m - model.previous_runs_from < timedelta(days=100):
                    q["first"] = next_month(m).isoformat()   # archive starts after this month
                    self.log.append(f"previous_runs/{name}: no data for {m:%Y-%m}, archive starts {q['first'][:7]}")
                    return "done"
                q.setdefault("empty", []).append(f"{m:%Y-%m}")  # a gap, not the start: keep going
                self.log.append(f"previous_runs/{name}: no data for {m:%Y-%m}")
                continue
            df["available_at"] = df["valid"] - pd.to_timedelta(df["lead_days"].astype("int64") * 24, unit="h") \
                + pd.Timedelta(hours=model.delay_h)
            df["fetched_at"] = pd.Timestamp(datetime.now(UTC)).floor("s")
            self.store.put_parquet(prev_path(name, m), df)
            self.files += 1
            self.checkpoint()
        return "done"

    # ------------------------------------------------------------------ single runs
    def single_runs(self, name: str, q: dict) -> str:
        model = MODELS[name]
        step = timedelta(hours=model.run_every_h)
        newest = self.t - timedelta(hours=12)
        newest = datetime(newest.year, newest.month, newest.day, tzinfo=UTC) + \
            step * ((newest.hour * 3600 + newest.minute * 60) // int(step.total_seconds()))
        first = datetime.fromisoformat(q["first"]) if q.get("first") else model.single_runs_from
        present = {p.rsplit("/", 1)[-1].split(".")[0] for p in self.store.list(f"weather/runs/{name}")}
        unavailable = set(q.get("unavailable", []))
        tried = {k: v for k, v in q.get("tried_empty", {}).items() if self.t - datetime.fromisoformat(v) < timedelta(hours=24)}
        q["tried_empty"] = tried
        streak = 0
        r = newest
        while r >= first:
            key = f"{r:%Y%m%dT%H}"
            if key in present or key in unavailable or key in tried:
                r -= step
                continue
            if not self.time_left():
                return "time"
            try:
                df = self.fetch(q, model, lambda v: self.client.single_run(model, r, POINTS, model.horizon_h, v))
            except openmeteo.BadRequest as exc:
                df = None
                self.log.append(f"single_runs/{name} {key}: {exc.reason[:120]}")
            if df is None or not _has_data(df, model.variables):
                if self.t - r > timedelta(days=7):   # old and empty: not in the archive, never ask again
                    unavailable.add(key)
                    q["unavailable"] = sorted(unavailable)
                else:                                # fresh runs may not be archived yet: retry tomorrow
                    tried[key] = self.t.isoformat()
                streak += 1
                # Documented archive start wrong? Only near it may a gap end the queue;
                # further inside, a gap is just a gap and older runs are still fetched.
                if streak >= EMPTY_STREAK_STOP and r - model.single_runs_from < timedelta(days=60):
                    q["first"] = (r + EMPTY_STREAK_STOP * step).isoformat()
                    self.log.append(f"single_runs/{name}: archive starts {q['first']}")
                    return "done"
                r -= step
                continue
            streak = 0
            df.insert(0, "run", pd.Timestamp(r))
            df.insert(1, "available_at", pd.Timestamp(r + timedelta(hours=model.delay_h)))
            df.insert(2, "fetched_at", pd.Timestamp(datetime.now(UTC)).floor("s"))
            df.insert(3, "source", "archive")
            self.store.put_parquet(run_path(name, r), df)
            present.add(key)
            self.files += 1
            self.checkpoint()
            r -= step
        return "done"

    def progress(self) -> dict:
        out = {}
        for kind, name in BACKFILL_ORDER:
            model = MODELS[name]
            q = self.state["queues"].get(f"{kind}/{name}", {})
            if kind == "previous_runs":
                have = len(self.store.list(f"weather/previous_runs/{name}"))
                out[f"{kind}/{name}"] = {"months": have, "status": q.get("status")}
            else:
                have = len(self.store.list(f"weather/runs/{name}"))
                span = self.t - (datetime.fromisoformat(q["first"]) if q.get("first") else model.single_runs_from)
                expected = int(span / timedelta(hours=model.run_every_h))
                out[f"{kind}/{name}"] = {"runs": have, "expected": expected, "status": q.get("status")}
        return out


def probe(client: openmeteo.Client) -> dict:
    """Which ECMWF run does `_previous_day1` come from? Compares against single runs."""
    model, pt = MODELS["ecmwf_ifs"], [p for p in POINTS if p.name == "BB"]
    day = datetime(2026, 5, 10, tzinfo=UTC)
    var = ("wind_speed_100m",)
    prev = client.previous_runs(model, pt, day, day + timedelta(days=1), (1,), var).set_index("valid")[var[0]]
    runs = [day - timedelta(hours=h) for h in range(0, 49, 6)]
    single = {r: client.single_run(model, r, pt, 72, var).set_index("valid")[var[0]] for r in runs}
    out = {}
    for valid, v in prev.items():
        match = [f"-{int((valid - r).total_seconds() // 3600)}h" for r, s in single.items()
                 if valid in s.index and np.isfinite(v) and abs(float(s[valid]) - float(v)) < 0.05]
        out[valid.strftime("%H:%M")] = match
    return out


def run(store: Store, max_s: float = 3000) -> int:
    t = now()
    state = store.read_json(STATE, default={}) or {}
    today = t.strftime("%Y-%m-%d")
    if state.get("day") != today:
        state["day"], state["day_used"] = today, 0.0
    state["_counted"] = 0.0
    state.setdefault("queues", {})
    budget = openmeteo.Budget(run_cap=OM_RUN_CAP, day_left=OM_DAILY_CAP - state["day_used"], minute_cap=OM_MINUTE_CAP)
    client = openmeteo.Client(budget)
    runner = Runner(store, state, client, t, max_s)
    stop_reason = "complete"
    errors: list[str] = []
    try:
        if "probe" not in state:
            try:
                state["probe"] = {"previous_day1_matches_run": probe(client), "at": iso(t)}
            except Exception as exc:
                errors.append(f"probe: {type(exc).__name__}: {str(exc)[:200]}")
        for kind, name in BACKFILL_ORDER:
            key = f"{kind}/{name}"
            q = state["queues"].setdefault(key, {})
            try:
                res = getattr(runner, kind)(name, q)
            except openmeteo.BudgetExhausted:
                stop_reason = "budget"
                break
            except openmeteo.RateLimited as exc:
                stop_reason = f"429 ({exc.scope})"
                errors.append(f"{key}: rate limited: {exc}")
                break
            except Exception as exc:
                errors.append(f"{key}: {type(exc).__name__}: {str(exc)[:200]}")
                LOG.error(traceback.format_exc())
                q["status"] = "error"
                continue
            q["status"] = res
            if res == "time":
                stop_reason = "time"
                break
    finally:
        runner.checkpoint(force=True)
        state["complete"] = stop_reason == "complete" and not errors
        state["progress"] = runner.progress()
        state["last"] = {"started": iso(t), "finished": iso(datetime.now(UTC)), "stop": stop_reason,
                         "files": runner.files, "calls": client.calls, "weight": round(client.weight_used, 1),
                         "errors": errors, "notes": runner.log[-30:]}
        state["day_used"] = round(state.get("day_used", 0) + client.weight_used - state.get("_counted", 0), 2)
        state.pop("_counted", None)
        store.put_json(STATE, state)
        store.commit(f"weather backfill {iso(t)}: {stop_reason}")
    LOG.info("weather backfill: %s", state["last"])
    return 0
