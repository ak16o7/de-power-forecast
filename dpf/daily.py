"""Daily job: consolidated ENTSO-E history, installed capacity, dataset card, health.

- Month files of A75 actuals and A69 day-ahead / intraday forecasts, DE + 4 control
  areas. The last 35 days are fetched again every day because TSOs replace
  provisional actuals with metered values.
- Missing months back to 2024-01 are backfilled, a time budget per run, resumable.
- Installed capacity (energy-charts, monthly) is kept as dated snapshots because
  it gets revised as late registrations come in.
- Health: fails the workflow (one e-mail a day at most) if the recorder or the
  weather backfill have been silent for too long.
"""
from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from . import entsoe
from .config import AREAS, ENERGY_CHARTS_CAPACITY, ENTSOE_HISTORY_START, USER_AGENT, UTC
from .store import Store
from .util import berlin_midnight, iso, months, next_month, now, parse_iso

LOG = logging.getLogger(__name__)

STATE = "state/daily.json"
STREAMS = {"a75": None, "a69_da": "A01", "a69_id": "A40"}
REFRESH_DAYS = 35


def entsoe_client() -> entsoe.Client:
    client = entsoe.Client()
    client.min_interval = 0.5   # <= 120 requests/min: the recorder shares the 400/min ENTSO-E limit
    return client


def month_path(stream: str, area: str, m: datetime) -> str:
    return f"entsoe/{stream}/{area}/{m:%Y}/{m:%Y-%m}.parquet"


def fetch_month(client: entsoe.Client, stream: str, area: str, m: datetime, until: datetime) -> pd.DataFrame:
    end = min(next_month(m), until)
    process = STREAMS[stream]
    df = client.actual(area, m, end) if process is None else client.forecast(area, process, m, end)
    df["fetched_at"] = pd.Timestamp(datetime.now(UTC)).floor("s")
    for c in ("area", "psr"):
        df[c] = df[c].astype("string")
    return df.sort_values(["psr", "ts"]).reset_index(drop=True)


def entsoe_months(store: Store, t: datetime, state: dict, client: entsoe.Client,
                  errors: list[str], budget_s: float) -> dict:
    t0 = time.monotonic()
    until = berlin_midnight(t, 2)  # end of tomorrow (Berlin): A69 day-ahead reaches into it
    recent = set(months(t - timedelta(days=REFRESH_DAYS), until))
    done = set(state.get("backfilled", []))
    todo = [(m, True) for m in sorted(recent)]
    todo += [(m, False) for m in reversed(months(ENTSOE_HISTORY_START, min(recent)))
             if f"{m:%Y-%m}" not in done]
    written, stopped = 0, False
    for m, is_recent in todo:
        if time.monotonic() - t0 > budget_s:
            stopped = True
            break
        month_ok = True
        for stream in STREAMS:
            for area in AREAS:
                try:
                    df = fetch_month(client, stream, area, m, until)
                except Exception as exc:
                    month_ok = False
                    errors.append(entsoe.mask(f"{stream}/{area}/{m:%Y-%m}: {type(exc).__name__}: {exc}")[:300])
                    continue
                if df.empty:
                    continue
                path = month_path(stream, area, m)
                old = store.read_parquet(path) if is_recent else None
                if old is not None:  # a short answer must not erase points we already have
                    df = (pd.concat([old, df], ignore_index=True)
                          .drop_duplicates(["area", "psr", "ts"], keep="last")
                          .sort_values(["psr", "ts"]).reset_index(drop=True))
                store.put_parquet(path, df)
                written += 1
        if month_ok and not is_recent:
            done.add(f"{m:%Y-%m}")
        if store.pending >= 30:
            state["backfilled"] = sorted(done)
            store.put_json(STATE, state)
            store.commit(f"entsoe months up to {m:%Y-%m}")
    state["backfilled"] = sorted(done)
    missing = [f"{m:%Y-%m}" for m in months(ENTSOE_HISTORY_START, min(recent)) if f"{m:%Y-%m}" not in done]
    return {"files_written": written, "requests": client.requests, "refreshed": sorted(f"{m:%Y-%m}" for m in recent),
            "backfill_missing": missing, "stopped_on_time_budget": stopped}


def capacity(store: Store, t: datetime) -> dict:
    r = requests.get(ENERGY_CHARTS_CAPACITY, params={"country": "de", "time_step": "monthly",
                                                     "installation_decommission": "false"},
                     headers={"User-Agent": USER_AGENT}, timeout=60)
    r.raise_for_status()
    j = r.json()
    month = pd.to_datetime(pd.Series(j["time"]), format="%m.%Y", utc=True)
    rows = []
    for pt in j["production_types"]:
        for m, v in zip(month, pt["data"]):
            if v is not None:
                rows.append((m, pt["name"], float(v)))
    df = pd.DataFrame(rows, columns=["month", "type", "gw"])
    df["type"] = df["type"].astype("string")
    prev = store.read_parquet("capacity/latest.parquet")
    same = prev is not None and prev[["month", "type", "gw"]].reset_index(drop=True).equals(df)
    if not same:
        snap = df.assign(seen_at=pd.Timestamp(t).floor("s"))
        store.put_parquet("capacity/latest.parquet", snap)
        store.put_parquet(f"capacity/snapshots/{t:%Y-%m-%d}.parquet", snap)
    return {"changed": not same, "types": sorted(df["type"].unique().tolist()), "last_month": str(month.max().date())}


def health(store: Store, t: datetime, since: datetime) -> dict:
    rec = store.read_json("state/record.json", default={}) or {}
    wb = store.read_json("state/weather_backfill.json", default={}) or {}
    rec_ok, wb_run = parse_iso(rec.get("last_ok")), parse_iso((wb.get("last") or {}).get("finished"))
    problems = []
    grace = t - since < timedelta(hours=6)   # first day: the other jobs may not have run yet
    if (rec_ok is None and not grace) or (rec_ok is not None and t - rec_ok > timedelta(hours=3)):
        problems.append(f"recorder: no successful run since {iso(rec_ok)}")
    if not wb.get("complete") and ((wb_run is None and not grace) or
                                   (wb_run is not None and t - wb_run > timedelta(hours=26))):
        problems.append(f"weather backfill: no run since {iso(wb_run)}")
    return {"recorder_last_ok": iso(rec_ok), "weather_backfill_last_run": iso(wb_run),
            "weather_backfill_progress": wb.get("progress"), "problems": problems}


def dataset_card(store: Store) -> bool:
    card = (Path(__file__).parent / "dataset_card.md").read_bytes()
    if store.read("README.md") == card:
        return False
    store.put("README.md", card)
    return True


def run(store: Store, budget_s: float = 1500) -> int:
    t = now()
    state = store.read_json(STATE, default={}) or {}
    errors: list[str] = []
    summary: dict = {"started": iso(t)}
    last_squash = parse_iso(state.get("last_squash"))
    if last_squash is None or t - last_squash > timedelta(days=30):
        # ~200 commits a day would otherwise keep every old version of every file forever
        try:
            if store.squash_history():
                summary["squashed"] = True
            state["last_squash"] = iso(t)
        except Exception as exc:
            errors.append(f"squash: {type(exc).__name__}: {str(exc)[:200]}")
    for name, fn in (("entsoe", lambda: entsoe_months(store, t, state, entsoe_client(), errors, budget_s)),
                     ("capacity", lambda: capacity(store, t)),
                     ("dataset_card", lambda: dataset_card(store))):
        try:
            summary[name] = fn()
        except Exception as exc:
            errors.append(entsoe.mask(f"{name}: {type(exc).__name__}: {exc}")[:300])
            LOG.error(entsoe.mask(traceback.format_exc()))
    state.setdefault("first_run", iso(t))
    summary["health"] = health(store, t, parse_iso(state["first_run"]))
    summary["finished"] = iso(datetime.now(UTC))
    summary["errors"] = errors
    state["last"] = summary
    store.put_json(STATE, state)
    store.put_json("status.json", {"updated": summary["finished"], "health": summary["health"],
                                   "entsoe_backfill_missing_months": len((summary.get("entsoe") or {}).get("backfill_missing", [])),
                                   "errors_today": errors})
    store.commit(f"daily {t:%Y-%m-%d}")
    for e in errors:
        LOG.warning(e)
    LOG.info("daily summary: %s", {k: v for k, v in summary.items() if k != "errors"})
    return 1 if summary["health"]["problems"] or len(errors) > 5 else 0
