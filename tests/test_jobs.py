from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import pytest

from dpf import daily, entsoe, openmeteo, record, weather_backfill
from dpf.config import AREAS, MODELS, POINTS
from dpf.store import LocalStore
from tests.fakes import FakeEntsoe, FakeOpenMeteo

UTC = timezone.utc
REAL_ENTSOE, REAL_OM = entsoe.Client, openmeteo.Client


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("ENTSOE_API_KEY", "test-key")
    fake_e = FakeEntsoe()
    fake_o = FakeOpenMeteo()
    monkeypatch.setattr(entsoe, "Client", lambda: REAL_ENTSOE(session=fake_e, min_interval=0))
    monkeypatch.setattr(openmeteo, "Client", lambda budget=None: REAL_OM(budget, session=fake_o))
    monkeypatch.setattr(openmeteo.time, "sleep", lambda s: None)
    monkeypatch.setattr(entsoe.time, "sleep", lambda s: None)
    return LocalStore(tmp_path), fake_e, fake_o, monkeypatch


def at(monkeypatch, iso):
    monkeypatch.setenv("DPF_NOW", iso)


# ---------------------------------------------------------------- recorder
def test_recorder_logs_only_changes_and_compacts(env):
    store, fe, fo, mp = env
    fo.meta_runs = {"dwd_icon_d2": datetime(2026, 9, 29, 9, tzinfo=UTC)}
    at(mp, "2026-09-29T12:07:00Z")
    assert record.run(store) == 0
    parts = store.list("vintages/entsoe/parts/2026-09-29")
    assert len(parts) == 1
    first = store.read_parquet(parts[0])
    assert set(first["stream"]) == {"a75", "a69_da", "a69_id", "a69_current"}
    assert set(first["area"]) == set(AREAS)
    assert set(first["psr"]) == {"solar", "wind_on", "wind_off"}      # B04 gas filtered out
    assert first["seen_at"].notna().all()

    # same data again: nothing new to log
    at(mp, "2026-09-29T12:22:00Z")
    record.run(store)
    assert len(store.list("vintages/entsoe/parts/2026-09-29")) == 2   # new quarter hour only
    second = store.read_parquet(store.list("vintages/entsoe/parts/2026-09-29")[1])
    assert set(second["stream"]) == {"a75"} and second["ts"].min() == pd.Timestamp("2026-09-29T12:15Z")

    # a revision of the day-ahead forecast is logged
    fe.bump["a69_da"] = 50
    at(mp, "2026-09-29T12:37:00Z")
    record.run(store)
    third = store.read_parquet(store.list("vintages/entsoe/parts/2026-09-29")[2])
    assert "a69_da" in set(third["stream"])

    # after midnight yesterday's parts are compacted into one file
    at(mp, "2026-09-30T00:07:00Z")
    record.run(store)
    assert store.list("vintages/entsoe/parts/2026-09-29") == []
    day = store.read_parquet("vintages/entsoe/2026/2026-09-29.parquet")
    assert len(day) == len(first) + len(second) + len(third)
    state = store.read_json("state/record.json")
    assert state["last"]["entsoe"]["compacted_days"] == ["2026-09-29"]


def test_recorder_saves_each_weather_run_once(env):
    store, fe, fo, mp = env
    run = datetime(2026, 9, 29, 9, tzinfo=UTC)
    fo.meta_runs = {"dwd_icon_d2": run}
    at(mp, "2026-09-29T12:07:00Z")
    record.run(store)
    path = "weather/runs/icon_d2/2026-09/20260929T09.parquet"
    df = store.read_parquet(path)
    assert len(df) == len(POINTS) * MODELS["icon_d2"].horizon_h
    assert (df["available_at"] == pd.Timestamp(run) + pd.Timedelta(seconds=5400)).all()
    assert df["source"].eq("live").all()
    n = len(fo.calls)
    at(mp, "2026-09-29T12:22:00Z")
    record.run(store)
    assert len(fo.calls) == n          # no new run, no API call
    state = store.read_json("state/record.json")
    assert any("ecmwf_ifs" in e for e in state["last"]["errors"])  # meta missing is reported, not fatal


def test_recorder_survives_a_failing_area(env):
    store, fe, fo, mp = env
    fe.fail.add(("A75", AREAS["TTG"]))
    at(mp, "2026-09-29T12:07:00Z")
    assert record.run(store) == 0
    df = store.read_parquet(store.list("vintages/entsoe/parts/2026-09-29")[0])
    assert "TTG" not in set(df.loc[df["stream"] == "a75", "area"])
    assert any("a75/TTG" in e for e in store.read_json("state/record.json")["last"]["errors"])


# ---------------------------------------------------------------- daily ENTSO-E
def test_daily_backfill_is_resumable_and_refreshes_recent(env):
    store, fe, fo, mp = env
    mp.setattr(daily, "ENTSOE_HISTORY_START", datetime(2026, 6, 1, tzinfo=UTC))
    mp.setattr(daily, "capacity", lambda s, t: {"skipped": True})
    at(mp, "2026-09-29T03:41:00Z")
    fe.no_data_areas.add(AREAS["TBW"])
    daily.run(store, budget_s=3600)
    st = store.read_json("state/daily.json")
    assert st["backfilled"] == ["2026-06", "2026-07"]
    assert st["last"]["entsoe"]["refreshed"] == ["2026-08", "2026-09"]
    df = store.read_parquet("entsoe/a75/DE/2026/2026-06.parquet")
    assert df["ts"].min() == pd.Timestamp("2026-06-01T00:00Z") and len(df) == 30 * 96 * 3
    assert store.read("entsoe/a75/TBW/2026/2026-06.parquet") is None
    assert store.read("README.md").startswith(b"---\nlicense: cc-by-4.0")
    calls = len(fe.calls)
    daily.run(store, budget_s=3600)
    assert len(fe.calls) - calls == 2 * 3 * len(AREAS)   # next run: only the recent months again
    status = store.read_json("status.json")
    assert status["health"]["problems"] == []      # grace period on the first day
    at(mp, "2026-09-29T12:00:00Z")
    daily.run(store, budget_s=0)
    assert len(store.read_json("status.json")["health"]["problems"]) == 2   # recorder + backfill never ran


# ---------------------------------------------------------------- weather backfill
def _wb(store, mp, iso, **caps):
    at(mp, iso)
    mp.setattr(weather_backfill, "OM_MINUTE_CAP", 1e9)   # sleep is patched out: never pace in tests
    for k, v in caps.items():
        mp.setattr(weather_backfill, k, v)
    weather_backfill.run(store, max_s=600)
    return store.read_json("state/weather_backfill.json")


def test_weather_backfill_previous_runs_then_single_runs(env):
    store, fe, fo, mp = env
    mp.setattr(weather_backfill, "BACKFILL_ORDER", (("previous_runs", "ecmwf_ifs"), ("single_runs", "icon_d2")))
    mp.setitem(MODELS, "ecmwf_ifs", replace(MODELS["ecmwf_ifs"], previous_runs_from=datetime(2026, 6, 1, tzinfo=UTC)))
    mp.setitem(MODELS, "icon_d2", replace(MODELS["icon_d2"], single_runs_from=datetime(2026, 9, 27, tzinfo=UTC)))
    st = _wb(store, mp, "2026-09-29T20:17:00Z", OM_RUN_CAP=10_000, OM_DAILY_CAP=10_000)
    assert st["probe"]["previous_day1_matches_run"]
    assert sorted(store.list("weather/previous_runs/ecmwf_ifs")) == [
        f"weather/previous_runs/ecmwf_ifs/2026/2026-0{m}.parquet" for m in (6, 7, 8)]
    prev = store.read_parquet("weather/previous_runs/ecmwf_ifs/2026/2026-08.parquet")
    lead = prev["valid"] - prev["available_at"]
    assert set(lead.dt.total_seconds() / 3600) == {24 - 8.5, 48 - 8.5}  # never later than the safe bound
    runs = store.list("weather/runs/icon_d2")
    # 2026-09-27 00:00 .. 2026-09-29 06:00 (newest = now - 12 h floored to the 3 h cycle)
    assert len(runs) == 19
    df = store.read_parquet(runs[0])
    assert df["source"].eq("archive").all()
    assert (df["available_at"] - df["run"]).eq(pd.Timedelta(hours=2.5)).all()
    assert st["complete"] is True and st["last"]["stop"] == "complete"
    calls = len(fo.calls)
    st = _wb(store, mp, "2026-09-29T20:47:00Z", OM_RUN_CAP=10_000, OM_DAILY_CAP=10_000)
    assert len(fo.calls) == calls        # nothing left to do, nothing fetched
    st = _wb(store, mp, "2026-09-29T21:17:00Z", OM_RUN_CAP=10_000, OM_DAILY_CAP=10_000)
    assert len(fo.calls) == calls + 2    # the 09:00 run is now 12 h old: fetched (onshore + offshore)


def test_weather_backfill_budget_and_resume(env):
    store, fe, fo, mp = env
    mp.setattr(weather_backfill, "BACKFILL_ORDER", (("single_runs", "icon_d2"),))
    mp.setitem(MODELS, "icon_d2", replace(MODELS["icon_d2"], single_runs_from=datetime(2026, 9, 25, tzinfo=UTC)))
    st = _wb(store, mp, "2026-09-29T20:17:00Z", OM_RUN_CAP=16 * 5 + 10, OM_DAILY_CAP=10_000)
    assert st["last"]["stop"] == "budget"
    assert len(store.list("weather/runs/icon_d2")) == 5
    assert st["day_used"] == pytest.approx(16 * 5 + 10)   # 10 = one-off probe
    st = _wb(store, mp, "2026-09-29T21:17:00Z", OM_RUN_CAP=16 * 5, OM_DAILY_CAP=16 * 8 + 10)
    assert len(store.list("weather/runs/icon_d2")) == 8   # daily cap hit after 3 more runs
    assert st["day_used"] == pytest.approx(16 * 8 + 10)
    st = _wb(store, mp, "2026-09-30T00:17:00Z", OM_RUN_CAP=10_000, OM_DAILY_CAP=10_000)
    assert st["day"] == "2026-09-30" and st["complete"] is True


def test_weather_backfill_handles_429_unknown_variable_and_archive_start(env):
    store, fe, fo, mp = env
    mp.setattr(weather_backfill, "BACKFILL_ORDER", (("single_runs", "icon_d2"),))
    mp.setitem(MODELS, "icon_d2", replace(MODELS["icon_d2"], single_runs_from=datetime(2026, 9, 20, tzinfo=UTC)))
    fo.reject_vars.add("snow_depth")
    fo.empty_before = datetime(2026, 9, 26, 0, tzinfo=UTC)
    st = _wb(store, mp, "2026-09-29T20:17:00Z", OM_RUN_CAP=10_000, OM_DAILY_CAP=10_000)
    q = st["queues"]["single_runs/icon_d2"]
    assert q["drop_vars"] == ["snow_depth"]
    assert q["first"].startswith("2026-09-26T00:00")
    df = store.read_parquet(store.list("weather/runs/icon_d2")[0])
    assert "snow_depth" not in df.columns
    fo.rate_limit_after = len(fo.calls)
    store.delete("weather/runs/icon_d2/")
    store.commit("wipe")
    st = _wb(store, mp, "2026-09-29T21:17:00Z", OM_RUN_CAP=10_000, OM_DAILY_CAP=10_000)
    assert st["last"]["stop"].startswith("429") and st["complete"] is False
