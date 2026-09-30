from datetime import timezone

import numpy as np
import pandas as pd

from dpf import model_id

UTC = timezone.utc
IDX = pd.date_range("2026-01-01", "2026-03-01", freq="15min", tz=UTC, inclusive="left")


def series(seed=0):
    rng = np.random.default_rng(seed)
    n = len(IDX)
    y = pd.Series(5000 + 3000 * np.sin(np.arange(n) / 50) + rng.normal(0, 50, n), index=IDX)
    slow_err = pd.Series(np.convolve(rng.normal(0, 60, n), np.ones(40), "same"), index=IDX)  # persistent error
    da = y - slow_err
    id_ = y - 0.6 * slow_err
    cap = pd.Series(50_000.0, index=IDX)
    return y, da, id_, cap


def test_rows_use_nothing_after_the_last_known_quarter_hour():
    y, da, id_, cap = series()
    issues = pd.DatetimeIndex([pd.Timestamp("2026-01-20 10:00", tz=UTC)])
    a = model_id.rows(y, da, id_, cap, issues, model_id.EVAL_K)
    later = y.copy()
    later[later.index > issues[0] - model_id.LAG * model_id.Q] += 1e6    # everything not yet known at T
    b = model_id.rows(later, da, id_, cap, issues, model_id.EVAL_K)
    pd.testing.assert_frame_equal(a[model_id.FEATURES], b[model_id.FEATURES])
    assert (a["y"] != b["y"]).all()                                          # only the target moved


def test_intraday_model_learns_the_persistent_tso_error(monkeypatch):
    monkeypatch.setattr(model_id, "ROUNDS", 80)
    y, da, id_, cap = series(1)
    hourly = pd.date_range("2026-01-02", "2026-02-01", freq="h", tz=UTC)
    test_issues = pd.date_range("2026-02-02", "2026-02-27", freq="15min", tz=UTC)
    model = model_id.fit(model_id.rows(y, da, id_, cap, hourly, model_id.TRAIN_K))
    test = model_id.rows(y, da, id_, cap, test_issues, (1, 4)).dropna(subset=["y", "tso_best"])
    err_model = np.abs(model_id.predict(model, test) - test["y"]).mean()
    err_tso = np.abs(test["tso_best"] - test["y"]).mean()
    assert err_model < 0.8 * err_tso


def test_paired_t_uses_days_and_detects_a_clear_difference():
    t = pd.date_range("2026-01-01", periods=96 * 60, freq="15min", tz=UTC)
    rng = np.random.default_rng(5)
    noise = rng.normal(0, 1, len(t))
    assert model_id.verdict(model_id.paired_t(pd.Series(t), 0.5 * noise, noise)) == "model better"
    other = rng.normal(0, 1, len(t))
    assert model_id.verdict(model_id.paired_t(pd.Series(t), other, noise)) == "not distinguishable"
    assert model_id.verdict(model_id.paired_t(pd.Series(t), noise, noise)) == "not distinguishable"
    assert model_id.verdict(model_id.paired_t(pd.Series(t[:96 * 3]), noise[:288], noise[:288])) == "too few days"
