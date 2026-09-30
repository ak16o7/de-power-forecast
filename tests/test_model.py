from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from dpf import baseline, features, model_da
from dpf.config import POINTS
from dpf.store import LocalStore

UTC = timezone.utc


def test_issue_time_is_11_local_on_the_day_before_across_dst():
    qh = pd.DatetimeIndex(["2026-03-29 12:00", "2026-10-25 23:45", "2026-10-25 01:30"], tz=UTC)
    local = features.issue_time(qh).tz_convert("Europe/Berlin")
    assert [str(t) for t in local] == ["2026-03-28 11:00:00+01:00", "2026-10-25 11:00:00+01:00",
                                       "2026-10-24 11:00:00+02:00"]


def test_solar_geometry():
    t = pd.date_range("2026-06-21", periods=96, freq="15min", tz=UTC)
    cz = features.cos_zenith(t, 51.2, 10.4)
    assert np.degrees(np.arcsin(cz.max())) == pytest.approx(62.2, abs=0.3)   # 90 - 51.2 + 23.4
    assert features.clear_sky_ghi(cz).min() == 0


def test_leak_check_refuses_weather_published_too_late():
    v = pd.date_range("2026-05-01", "2026-05-05", freq="h", tz=UTC)
    qh = pd.date_range("2026-05-02", "2026-05-04", freq="15min", tz=UTC, inclusive="left")
    ok = pd.DataFrame({"valid": v, "available_at": v - pd.Timedelta(hours=43.5)})
    features.availability_check(ok, qh)
    late = ok.assign(available_at=v - pd.Timedelta(hours=19.5))       # a day-1 forecast
    with pytest.raises(RuntimeError, match="after their issue time"):
        features.availability_check(late, qh)


def test_radiation_is_placed_mid_hour_and_gaps_are_not_bridged():
    v = pd.date_range("2026-05-01 00:00", periods=10, freq="h", tz=UTC)
    w = pd.DataFrame({"valid": v, "point": "BB", "shortwave_radiation": np.arange(10) * 100.0,
                      "temperature_2m": np.arange(10) * 1.0})
    w.loc[5, ["shortwave_radiation", "temperature_2m"]] = np.nan   # 1 h hole: bridged
    w2 = w.copy()
    w2.loc[2:6, ["shortwave_radiation", "temperature_2m"]] = np.nan   # 5 h hole: not bridged
    qh = pd.date_range("2026-05-01 01:00", periods=4, freq="15min", tz=UTC)
    out = features.to_quarter_hours(w, qh)
    # value at 01:00 is the mean of 00:00-01:00, centred 00:30; QH 01:00-01:15 is centred 01:07:30
    assert out["BB__shortwave_radiation"].iloc[0] == pytest.approx(100 + 100 * 37.5 / 60)
    assert out["BB__temperature_2m"].iloc[0] == pytest.approx(1 + 7.5 / 60)
    qh2 = pd.date_range("2026-05-01 04:00", periods=4, freq="15min", tz=UTC)
    assert features.to_quarter_hours(w2, qh2)["BB__temperature_2m"].isna().all()


def _synthetic_store(tmp_path, months=5):
    """Wind speed drives wind output, clear sky drives solar: a learnable toy world."""
    store = LocalStore(tmp_path)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    v = pd.date_range(start - pd.Timedelta(days=1), periods=24 * 31 * months + 48, freq="h", tz=UTC)
    rng = np.random.default_rng(0)
    ws = 20 + 10 * np.sin(np.arange(len(v)) / 17) + rng.normal(0, 1, len(v))
    rows = []
    for p in POINTS:
        rows.append(pd.DataFrame({"point": p.name, "valid": v, "lead_days": 2,
                                  "available_at": v - pd.Timedelta(hours=43.5), "fetched_at": v,
                                  "shortwave_radiation": features.clear_sky_ghi(features.cos_zenith(v, p.lat, p.lon)),
                                  "direct_radiation": 0.0, "diffuse_radiation": 0.0, "temperature_2m": 10.0,
                                  "cloud_cover": 0.0, "wind_speed_120m": ws, "wind_direction_120m": 270.0,
                                  "wind_gusts_10m": ws * 1.5, "surface_pressure": 1013.0}))
    w = pd.concat(rows)
    for m, g in w.groupby(w["valid"].dt.strftime("%Y-%m")):
        store.put_parquet(f"weather/previous_runs/icon_eu/{m[:4]}/{m}.parquet", g)
    qh = pd.date_range(start, periods=96 * 31 * months, freq="15min", tz=UTC)
    wsq = np.interp(qh.asi8, v.asi8, ws)
    cs = features.clear_sky_ghi(features.cos_zenith(qh + pd.Timedelta(minutes=7.5), 51.2, 10.4))
    mw = {"solar": cs * 50, "wind_on": wsq ** 2 * 50, "wind_off": wsq * 200}
    a = pd.DataFrame({"ts": np.repeat(qh, 3), "psr": np.tile(list(mw), len(qh)),
                      "mw": np.column_stack([mw[k] for k in mw]).ravel(), "area": "DE"})
    for stream, add in (("a75", 0.0), ("a69_da", 500.0)):
        d = a.assign(mw=a["mw"] + add)
        for m, g in d.groupby(d["ts"].dt.strftime("%Y-%m")):
            store.put_parquet(f"entsoe/{stream}/DE/{m[:4]}/{m}.parquet", g)
    cap = pd.DataFrame({"month": pd.date_range("2025-09-01", periods=12, freq="MS", tz=UTC)})
    cap = pd.concat([cap.assign(type=n, gw=100.0) for n in baseline.CAPACITY_TYPE.values()])
    store.put_parquet("capacity/latest.parquet", cap)
    store.commit("synthetic")
    return store


def test_walk_forward_learns_and_never_sees_the_test_month(tmp_path, monkeypatch):
    monkeypatch.setattr(model_da, "PARAMS", dict(model_da.PARAMS, n_estimators=60))
    monkeypatch.setattr(model_da, "TRAIN_FROM", datetime(2026, 1, 1, tzinfo=UTC))
    store = _synthetic_store(tmp_path)
    test = [datetime(2026, 4, 1, tzinfo=UTC)]
    pred = model_da.backtest(store, test)
    assert set(pred["tech"]) == {"solar", "wind_on", "wind_off"}
    assert (pred["p10"] <= pred["p50"]).all() and (pred["p50"] <= pred["p90"]).all()
    cap = baseline.capacity_mw(store, pd.DatetimeIndex(pred["ts"].unique()))
    scores = {r["tech"]: r for r in model_da.score(pred, cap)}
    assert scores["wind_off"]["model_mae_mw"] < scores["wind_off"]["tso_mae_mw"]   # TSO is off by 500 MW
    # poison the actuals of the test month: predictions must not move
    for f in store.list("entsoe/a75/DE"):
        if f.endswith("2026-04.parquet"):
            df = store.read_parquet(f)
            store.put_parquet(f, df.assign(mw=df["mw"] * 10))
    store.commit("poison")
    again = model_da.backtest(store, test)
    np.testing.assert_allclose(again["p50"].to_numpy(), pred["p50"].to_numpy())


def test_widening_reaches_the_target_coverage():
    rng = np.random.default_rng(3)
    actual = pd.Series(rng.normal(0.5, 0.1, 5000))
    narrow = pd.DataFrame({"p10": 0.5 - 0.05, "p50": 0.5, "p90": 0.5 + 0.05}, index=actual.index)
    k = model_da.widening(narrow, actual)
    wide = model_da.widen(narrow, k)
    inside = ((actual >= wide["p10"]) & (actual <= wide["p90"])).mean()
    assert k > 2 and inside == pytest.approx(0.8, abs=0.01)
