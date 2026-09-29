import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from dpf import baseline
from dpf.store import LocalStore

UTC = timezone.utc
IDX = pd.date_range("2026-01-01", "2026-05-01", freq="15min", tz=UTC, inclusive="left")


def frames(actual_values):
    actual = pd.DataFrame({t: actual_values for t in baseline.TECHS}, index=IDX)
    return actual, actual + 100.0, actual + 50.0    # actual, TSO day-ahead, TSO intraday


def test_day_ahead_and_persistence_errors():
    ramp = np.arange(len(IDX), dtype=float)          # +1 MW every quarter hour
    actual, da, id_ = frames(ramp)
    start, end = datetime(2026, 3, 1, tzinfo=UTC), datetime(2026, 4, 1, tzinfo=UTC)
    errs = baseline.errors(actual, da, id_, pd.date_range(start, end, freq="15min", inclusive="left"))
    cap = pd.DataFrame(10_000.0, index=IDX, columns=list(baseline.TECHS))
    res = baseline.summarize(errs, cap, start, end)
    da_rows = {r["benchmark"]: r for r in res["day_ahead"] if r["tech"] == "solar"}
    assert da_rows["tso_da"]["mae_mw"] == 100 and da_rows["tso_da"]["bias_mw"] == 100
    assert da_rows["tso_da"]["nmae_pct"] == 1.0
    assert da_rows["naive_d2"]["bias_mw"] == -192                    # two days = 192 quarter hours
    pers = {r["lead_min"]: r for r in res["intraday"] if r["tech"] == "solar" and r["benchmark"] == "persistence"}
    for k in baseline.HORIZONS_QH:                                     # known one hour after the quarter hour ends
        assert pers[15 * k]["bias_mw"] == -(k + baseline.KNOWN_AFTER_QH)
    last = {r["lead_min"]: r for r in res["intraday"] if r["tech"] == "solar" and r["benchmark"] == "tso_last_error"}
    assert all(r["mae_mw"] == 0 for r in last.values())               # constant TSO offsets are fully corrected
    ns = {r["n"] for r in res["intraday"] if r["tech"] == "solar"}
    assert len(ns) == 1                                                # same targets for all benchmarks and leads


def test_tso_intraday_only_after_its_publication():
    actual, da, id_ = frames(np.zeros(len(IDX)))
    target = pd.date_range("2026-03-10", "2026-03-11", freq="15min", tz="Europe/Berlin", inclusive="left").tz_convert(UTC)
    for k in (1, 8, 32):
        best = baseline.tso_best(da["solar"].reindex(target), id_["solar"], k)
        issue_local = (target - pd.Timedelta(minutes=15 * k)).tz_convert("Europe/Berlin")
        same_day_after_8 = (issue_local.day == 10) & (issue_local.hour >= 8)
        assert (best[same_day_after_8] == 50).all()                   # intraday forecast
        before = ~same_day_after_8 & ((issue_local.day == 10) | (issue_local.hour >= 18))
        assert (best[before] == 100).all()                             # only day-ahead was out
        assert best[~same_day_after_8 & ~before].isna().all()          # nothing published yet


def test_publication_rule_holds_across_dst():
    target = pd.DatetimeIndex([pd.Timestamp("2026-03-29 08:15", tz="Europe/Berlin").tz_convert(UTC)])
    day_diff, hour = baseline.local_hour_of_issue(target, 1)
    assert day_diff[0] == 0 and hour[0] == 8.0


def test_report_job_writes_json(tmp_path, monkeypatch):
    store = LocalStore(tmp_path)
    t = IDX
    base = pd.DataFrame({"ts": np.repeat(t, 3), "psr": np.tile(list(baseline.TECHS), len(t))})
    for stream, add in (("a75", 0.0), ("a69_da", 100.0), ("a69_id", 50.0)):
        df = base.assign(area="DE", mw=1000.0 + add)
        for m, g in df.groupby(df["ts"].dt.strftime("%Y-%m")):
            store.put_parquet(f"entsoe/{stream}/DE/{m[:4]}/{m}.parquet", g)
    cap = pd.DataFrame({"month": pd.date_range("2025-12-01", "2026-06-01", freq="MS", tz=UTC)})
    cap = pd.concat([cap.assign(type=n, gw=10.0) for n in baseline.CAPACITY_TYPE.values()])
    store.put_parquet("capacity/latest.parquet", cap)
    store.commit("fixtures")
    monkeypatch.setenv("DPF_NOW", "2026-04-15T04:00:00Z")
    assert baseline.run(store) == 0
    rep = json.loads(store.read(baseline.REPORT))
    assert rep["period"] == {"from": "2026-02-01T00:00:00Z", "to": "2026-04-01T00:00:00Z"}
    row = next(r for r in rep["all"]["day_ahead"] if r["tech"] == "wind_on" and r["benchmark"] == "tso_da")
    assert row["mae_mw"] == 100 and row["nmae_pct"] == pytest.approx(1.0)
    assert {r["month"] for r in rep["monthly"]} == {"2026-02", "2026-03"}
    bar = {(r["product"], r["tech"], r.get("lead_min")): r["benchmark"] for r in rep["bar_last_12"]}
    assert bar[("day_ahead", "solar", None)] in ("naive_d2", "clim14")     # flat actuals: naive rules are exact
    assert bar[("intraday", "solar", 60)] in ("persistence", "tso_last_error")


def test_last_error_uses_only_known_values():
    rng = np.random.default_rng(1)
    actual, da, id_ = frames(rng.normal(1000, 100, len(IDX)))
    y = actual["wind_on"]
    target = IDX[(IDX >= "2026-03-01") & (IDX < "2026-03-03")]
    k = 4
    fc = baseline.tso_last_error(y, da["wind_on"], id_["wind_on"], target, k)
    moved = y.copy()
    moved[moved.index > target[0] - pd.Timedelta(minutes=15 * (k + baseline.KNOWN_AFTER_QH))] += 1e6
    fc2 = baseline.tso_last_error(moved, da["wind_on"], id_["wind_on"], target[:1], k)
    assert fc2.iloc[0] == pytest.approx(fc.iloc[0])      # later actuals cannot change the forecast
