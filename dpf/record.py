"""Recorder, every 15 minutes: what did the world look like right now?

1. ENTSO-E actuals (A75, last 6 h) and TSO forecasts (A69 day-ahead, intraday,
   current) for DE and the four control areas. Only new or changed values are
   appended to a vintage log, each with the time we saw it (`seen_at`).
2. Every new weather model run (checked via Open-Meteo's run metadata) is saved
   once, with the time Open-Meteo made it available.
"""
from __future__ import annotations

import logging
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd

from . import entsoe, openmeteo
from .config import A69_PROCESSES, AREAS, MODELS, POINTS, UTC
from .store import Store
from .util import berlin_midnight, floor, iso, now, parse_iso
from .vintage import changes, latest

LOG = logging.getLogger(__name__)

KEYS = ["stream", "area", "psr", "ts"]
LOG_COLUMNS = KEYS + ["mw", "seen_at"]
PARTS = "vintages/entsoe/parts"
STATE = "state/record.json"


def day_path(day: str) -> str:
    return f"vintages/entsoe/{day[:4]}/{day}.parquet"


def run_path(model: str, run: datetime) -> str:
    return f"weather/runs/{model}/{run:%Y-%m}/{run:%Y%m%dT%H}.parquet"


def read_parts(store: Store, day: str) -> list[pd.DataFrame]:
    with ThreadPoolExecutor(8) as pool:  # up to 96 small files per day
        parts = list(pool.map(store.read_parquet, store.list(f"{PARTS}/{day}")))
    return [p for p in parts if p is not None and not p.empty]


def load_day(store: Store, day: str) -> pd.DataFrame | None:
    """Vintage log of one UTC day: the compacted file, or the parts of a day in progress."""
    df = store.read_parquet(day_path(day))
    if df is not None:
        return df
    parts = read_parts(store, day)
    return pd.concat(parts, ignore_index=True) if parts else None


def compact(store: Store, today: str) -> list[str]:
    """Merge the parts of finished days into one file per day."""
    done = []
    days = sorted({p.split("/")[3] for p in store.list(PARTS)})
    for day in days:
        if day >= today:
            continue
        parts = read_parts(store, day)
        old = store.read_parquet(day_path(day))
        frames = ([old] if old is not None else []) + parts
        if frames:
            merged = pd.concat(frames, ignore_index=True).drop_duplicates().sort_values(["seen_at", *KEYS])
            store.put_parquet(day_path(day), merged.reset_index(drop=True))
        store.delete(f"{PARTS}/{day}/")
        done.append(day)
    return done


def fetch_entsoe(client: entsoe.Client, t: datetime, errors: list[str]) -> pd.DataFrame:
    frames = []
    a75_start, a75_end = floor(t, 15) - timedelta(hours=6), floor(t, 15) + timedelta(minutes=15)
    a69_start, a69_end = berlin_midnight(t), berlin_midnight(t, 2)
    jobs = [("a75", None, a75_start, a75_end)] + [(s, p, a69_start, a69_end) for p, s in A69_PROCESSES.items()]
    for stream, process, start, end in jobs:
        for area in AREAS:
            try:
                df = client.actual(area, start, end) if process is None else client.forecast(area, process, start, end)
            except Exception as exc:  # one failing area must not stop the rest
                errors.append(entsoe.mask(f"{stream}/{area}: {type(exc).__name__}: {exc}")[:300])
                continue
            if df.empty:
                continue
            df.insert(0, "stream", stream)
            df["seen_at"] = pd.Timestamp(datetime.now(UTC)).floor("s")
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=LOG_COLUMNS)
    out = pd.concat(frames, ignore_index=True)[LOG_COLUMNS]
    for c in ("stream", "area", "psr"):
        out[c] = out[c].astype("string")
    out["mw"] = out["mw"].astype("float64")
    return out


def record_entsoe(store: Store, t: datetime, client: entsoe.Client, errors: list[str]) -> dict:
    today, yesterday = t.strftime("%Y-%m-%d"), (t - timedelta(days=1)).strftime("%Y-%m-%d")
    compacted = compact(store, today)
    known_frames = [f for f in (load_day(store, yesterday), load_day(store, today)) if f is not None]
    known = latest(pd.concat(known_frames, ignore_index=True), KEYS) if known_frames else None
    fresh = fetch_entsoe(client, t, errors)
    new = changes(fresh, known, KEYS)
    if not new.empty:
        store.put_parquet(f"{PARTS}/{today}/{t:%H%M%S}.parquet", new)
    return {"fetched_rows": int(len(fresh)), "logged_rows": int(len(new)), "compacted_days": compacted,
            "requests": client.requests,
            "logged_by_stream": {k: int(v) for k, v in new.groupby("stream").size().items()} if len(new) else {}}


def record_weather(store: Store, state: dict, client: openmeteo.Client, errors: list[str]) -> dict:
    """Save every new model run once (latest run via the forecast API)."""
    saved = {}
    runs = state.setdefault("last_run", {})
    for name, model in MODELS.items():
        meta = client.meta(model)
        if not meta or "last_run_initialisation_time" not in meta:
            errors.append(f"meta {name}: unavailable ({getattr(client, 'meta_error', None)})")
            continue
        run = datetime.fromtimestamp(meta["last_run_initialisation_time"], UTC)
        available = datetime.fromtimestamp(meta.get("last_run_availability_time", 0), UTC)
        prev = parse_iso(runs.get(name))
        if prev is not None and run <= prev:
            continue
        try:
            df = client.forecast(model, POINTS, run, model.horizon_h)
        except Exception as exc:
            errors.append(f"weather {name} {iso(run)}: {type(exc).__name__}: {str(exc)[:200]}")
            continue
        fetched = pd.Timestamp(datetime.now(UTC)).floor("s")
        # Guard against a newer run landing between the metadata check and the download.
        after = client.meta(model)
        if after and after.get("last_run_initialisation_time") != meta["last_run_initialisation_time"]:
            errors.append(f"weather {name}: run changed during download, retry next time")
            continue
        df.insert(0, "run", pd.Timestamp(run))
        df.insert(1, "available_at", pd.Timestamp(available))
        df.insert(2, "fetched_at", fetched)
        df.insert(3, "source", "live")
        store.put_parquet(run_path(name, run), df)
        runs[name] = iso(run)
        saved[name] = iso(run)
    return {"saved_runs": saved, "calls": client.calls, "weight": round(client.weight_used, 1)}


def run(store: Store) -> int:
    t = now()
    state = store.read_json(STATE, default={}) or {}
    errors: list[str] = []
    summary: dict = {"started": iso(t)}
    ok = False
    try:
        try:
            client = entsoe.Client()
            client.min_interval = 0.3
            summary["entsoe"] = record_entsoe(store, t, client, errors)
            ok = ok or summary["entsoe"]["fetched_rows"] > 0
        except Exception as exc:
            errors.append(entsoe.mask(f"entsoe: {type(exc).__name__}: {exc}")[:300])
            LOG.error(entsoe.mask(traceback.format_exc()))
        try:
            budget = openmeteo.Budget(run_cap=200, day_left=1e9, minute_cap=200)
            summary["weather"] = record_weather(store, state, openmeteo.Client(budget), errors)
            ok = ok or bool(summary["weather"]["saved_runs"])
        except Exception as exc:
            errors.append(f"weather: {type(exc).__name__}: {str(exc)[:200]}")
            LOG.error(traceback.format_exc())
    finally:
        summary["finished"] = iso(datetime.now(UTC))
        summary["errors"] = errors
        state["last"] = summary
        if ok:
            state["last_ok"] = summary["finished"]
        state["runs"] = int(state.get("runs", 0)) + 1
        store.put_json(STATE, state)
        store.commit(f"record {iso(t)}")
    for e in errors:
        LOG.warning(e)
    LOG.info("record summary: %s", {k: v for k, v in summary.items() if k != "errors"})
    # Upstream hiccups are logged in state/record.json; the daily job alarms if
    # nothing succeeded for hours, so a single bad quarter hour sends no e-mail.
    return 0
