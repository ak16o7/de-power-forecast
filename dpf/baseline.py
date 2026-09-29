"""The bar to beat: how good are the TSO forecasts and simple rules, per technology and horizon?

Day-ahead (every quarter hour of day D):
- tso_da: TSO day-ahead forecast (ENTSO-E A69/A01). Published 18:00 CET on D-1, i.e. later
  than a real day-ahead forecast has to be ready (before the 12:00 auction). Advantage TSO.
- naive_d2: the same quarter hour two days earlier (the last complete day known on D-1 morning).
- clim14: mean of the same quarter hour over D-15 .. D-2.

Intraday (issued every 15 min, target starts k quarter hours later, k = 1 .. 32):
- tso_best: the newest TSO forecast published at issue time: intraday (A40, 08:00 on D)
  if available, else day-ahead (A01, 18:00 on D-1). The TSO "current" forecast (A18) keeps
  no history and cannot be backtested; the recorder collects it from now on.
- tso_last_error: tso_best plus that forecast's error on the last known quarter hour.
- persistence: the last actual known at issue time, held flat. Actuals count as known one
  hour after their quarter hour ends (measured lag for Germany: 22-50 min).

Honest limits, written into the report: history uses today's (metered) actuals, both as
target and as persistence input; the recorder measures how much the first values differ.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from .config import UTC
from .store import Store
from .util import BERLIN, iso, month_start, now

LOG = logging.getLogger(__name__)

TECHS = ("solar", "wind_on", "wind_off")
CAPACITY_TYPE = {"solar": "Solar AC", "wind_on": "Wind onshore", "wind_off": "Wind offshore"}
HORIZONS_QH = (1, 2, 4, 8, 16, 32)          # 15 min, 30 min, 1 h, 2 h, 4 h, 8 h ahead
KNOWN_AFTER_QH = 5                           # a quarter hour is known 1 h after it ends
ID_PUBLISHED = 8                             # A40: 08:00 local on D
DA_PUBLISHED = 18                            # A01: 18:00 local on D-1
REPORT = "reports/baseline.json"


def load(store: Store, stream: str, area: str = "DE") -> pd.DataFrame:
    """Wide frame: UTC quarter-hour index, one column per technology."""
    frames = [store.read_parquet(p) for p in store.list(f"entsoe/{stream}/{area}")]
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame(columns=list(TECHS))
    df = pd.concat(frames, ignore_index=True)
    wide = df.pivot_table(index="ts", columns="psr", values="mw", aggfunc="last")
    idx = pd.date_range(wide.index.min(), wide.index.max(), freq="15min")
    return wide.reindex(idx).reindex(columns=list(TECHS))


def capacity_mw(store: Store, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Installed capacity (MW) of the month each quarter hour falls in."""
    cap = store.read_parquet("capacity/latest.parquet")
    out = pd.DataFrame(index=index, columns=list(TECHS), dtype="float64")
    if cap is None:
        return out
    utc = index.tz_convert(UTC)
    months = (utc - pd.to_timedelta(utc.day - 1, unit="D")).normalize()
    for tech, name in CAPACITY_TYPE.items():
        series = cap[cap["type"] == name].set_index("month")["gw"] * 1000
        out[tech] = series.reindex(months).to_numpy()
    return out


def _metrics(err: pd.Series, cap: pd.Series) -> dict:
    e = err.dropna()
    if e.empty:
        return {"n": 0}
    c = cap.reindex(e.index)
    return {"n": int(len(e)), "mae_mw": round(float(e.abs().mean()), 1),
            "rmse_mw": round(float(np.sqrt((e ** 2).mean())), 1), "bias_mw": round(float(e.mean()), 1),
            "nmae_pct": round(float((e.abs() / c).mean() * 100), 2) if c.notna().all() else None}


def local_hour_of_issue(target: pd.DatetimeIndex, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For target quarter hours and a lead of k quarter hours: issue time's local date offset
    to the target's local day (0 = same day, -1 = day before, ...) and local hour of the issue."""
    t_local = target.tz_convert(BERLIN)
    issue_local = (target - pd.Timedelta(minutes=15 * k)).tz_convert(BERLIN)
    day_diff = (issue_local.normalize().tz_localize(None) - t_local.normalize().tz_localize(None)).days
    hour = issue_local.hour + issue_local.minute / 60
    return np.asarray(day_diff), np.asarray(hour)


def _published(index: pd.DatetimeIndex, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Which TSO forecast of each target was out k quarter hours before it: (intraday, day-ahead only)."""
    day_diff, hour = local_hour_of_issue(index, k)
    use_id = (day_diff == 0) & (hour >= ID_PUBLISHED)
    use_da = ~use_id & ((day_diff == 0) | ((day_diff == -1) & (hour >= DA_PUBLISHED)))
    return use_id, use_da


def tso_best(da: pd.Series, id_: pd.Series, k: int) -> pd.Series:
    """TSO forecast for each target that was already published k quarter hours before it."""
    use_id, use_da = _published(da.index, k)
    out = pd.Series(np.nan, index=da.index)
    out[use_id] = id_.reindex(da.index)[use_id]
    out[use_da] = da[use_da]
    return out


def tso_last_error(y: pd.Series, da: pd.Series, id_: pd.Series, index: pd.DatetimeIndex, k: int) -> pd.Series:
    """Newest TSO forecast plus its error on the last known quarter hour (same TSO product)."""
    use_id, use_da = _published(index, k)
    lag = k + KNOWN_AFTER_QH
    err_id = (y - id_.reindex(y.index)).shift(lag).reindex(index)
    err_da = (y - da.reindex(y.index)).shift(lag).reindex(index)
    out = pd.Series(np.nan, index=index)
    out[use_id] = (id_.reindex(index) + err_id)[use_id]
    out[use_da] = (da.reindex(index) + err_da)[use_da]
    return out


def errors(actual: pd.DataFrame, da: pd.DataFrame, id_: pd.DataFrame, idx: pd.DatetimeIndex) -> dict:
    """forecast - actual for every benchmark, on one common set of targets per product and
    technology, so that benchmarks and horizons are compared on exactly the same quarter hours."""
    a = actual.reindex(actual.index.union(idx))
    out: dict[tuple, pd.Series] = {}
    for tech in TECHS:
        y = a[tech]
        target = y.reindex(idx)
        clim = sum(y.shift(96 * d) for d in range(2, 16)) / 14
        day_ahead = {"tso_da": da[tech].reindex(idx), "naive_d2": y.shift(192).reindex(idx),
                     "clim14": clim.reindex(idx)}
        errs = {("day_ahead", tech, name, None): fc - target for name, fc in day_ahead.items()}
        common = np.logical_and.reduce([e.notna().to_numpy() for e in errs.values()])
        out.update({k: e.where(common) for k, e in errs.items()})
        errs = {}
        for k in HORIZONS_QH:
            errs[("intraday", tech, "tso_best", 15 * k)] = tso_best(da[tech].reindex(idx), id_[tech], k) - target
            errs[("intraday", tech, "tso_last_error", 15 * k)] = tso_last_error(y, da[tech], id_[tech], idx, k) - target
            errs[("intraday", tech, "persistence", 15 * k)] = y.shift(k + KNOWN_AFTER_QH).reindex(idx) - target
        common = np.logical_and.reduce([e.notna().to_numpy() for e in errs.values()])
        out.update({k: e.where(common) for k, e in errs.items()})
    return out


def summarize(errs: dict, cap: pd.DataFrame, start: datetime, end: datetime) -> dict:
    out = {"day_ahead": [], "intraday": []}
    for (product, tech, bench, lead), e in errs.items():
        row = {"tech": tech, "benchmark": bench, **({"lead_min": lead} if lead else {}),
               **_metrics(e[(e.index >= start) & (e.index < end)], cap[tech])}
        out[product].append(row)
    return out


def bar(summary: dict) -> list[dict]:
    """Best benchmark per product, technology and lead time: the number our model has to beat."""
    best: dict[tuple, dict] = {}
    for product, rows in summary.items():
        for r in rows:
            key = (product, r["tech"], r.get("lead_min"))
            if r.get("n") and (key not in best or r["mae_mw"] < best[key]["mae_mw"]):
                best[key] = {"product": product, **r}
    return list(best.values())


def monthly(errs: dict, cap: pd.DataFrame) -> list[dict]:
    rows = []
    for (product, tech, bench, lead), e in errs.items():
        if product == "intraday" and lead not in (60, 240):
            continue
        df = pd.DataFrame({"abs": e.abs(), "rel": e.abs() / cap[tech].reindex(e.index)}).dropna()
        month = df.index.tz_convert(UTC).strftime("%Y-%m")
        for m, g in df.groupby(month):
            rows.append({"month": m, "product": product, "tech": tech, "benchmark": bench,
                         **({"lead_min": lead} if lead else {}),
                         "mae_mw": round(float(g["abs"].mean()), 1), "nmae_pct": round(float(g["rel"].mean() * 100), 2)})
    return rows


def run(store: Store) -> int:
    t = now()
    actual, da, id_ = load(store, "a75"), load(store, "a69_da"), load(store, "a69_id")
    if actual.empty or da.empty:
        LOG.error("no ENTSO-E data yet")
        return 1
    end = month_start(t)                                   # complete months only
    start = max(actual.index.min() + timedelta(days=16), da.index.min())
    start = month_start(start) if start == month_start(start) else \
        datetime(start.year + (start.month == 12), start.month % 12 + 1, 1, tzinfo=UTC)
    cap = capacity_mw(store, pd.date_range(start, end, freq="15min", inclusive="left"))
    last12 = datetime(end.year - 1, end.month, 1, tzinfo=UTC)
    report = {
        "generated_at": iso(t),
        "area": "DE",
        "period": {"from": iso(start), "to": iso(end)},
        "last_12_months": {"from": iso(max(start, last12)), "to": iso(end)},
        "method": {
            "error": "forecast minus actual, MW, every quarter hour (nights included for solar)",
            "targets": "per product and technology all benchmarks and horizons are scored on the same quarter hours",
            "solar_actual": "German solar 'actuals' are TSO extrapolations from reference plants, which flatters TSO solar forecasts",
            "nmae": "MAE / installed capacity of that month (Energy-Charts)",
            "actual_known_after_min": 15 * KNOWN_AFTER_QH,
            "tso_da_published": "18:00 local on D-1 (later than a real day-ahead deadline: advantage TSO)",
            "tso_id_published": "08:00 local on D",
            "not_backtestable": "TSO 'current' forecast (A18): only its last version is archived",
            "limits": "history uses today's metered actuals as target and as persistence input",
        },
    }
    errs = errors(actual, da, id_, pd.date_range(start, end, freq="15min", inclusive="left"))
    report["all"] = summarize(errs, cap, start, end)
    report["last_12"] = summarize(errs, cap, max(start, last12), end)
    report["bar_last_12"] = bar(report["last_12"])
    report["monthly"] = monthly(errs, cap)
    store.put_json(REPORT, report)
    store.commit(f"baseline report {t:%Y-%m-%d}")
    for r in report["bar_last_12"]:
        LOG.info("BAR %-9s %-8s %4s min  %-15s MAE %7.1f MW  nMAE %5s %%", r["product"], r["tech"],
                 r.get("lead_min", "-"), r["benchmark"], r["mae_mw"], r.get("nmae_pct"))
    for r in report["last_12"]["day_ahead"]:
        LOG.info("DA  %-8s %-11s MAE %7.1f MW  nMAE %5s %%", r["tech"], r["benchmark"], r.get("mae_mw", float("nan")), r.get("nmae_pct"))
    for r in report["last_12"]["intraday"]:
        LOG.info("ID  %-8s %-11s %3d min  MAE %7.1f MW  nMAE %5s %%", r["tech"], r["benchmark"], r["lead_min"],
                 r.get("mae_mw", float("nan")), r.get("nmae_pct"))
    return 0
