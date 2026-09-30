"""Intraday model v2: corrects the newest published TSO forecast, 15 min to 8 h ahead.

At issue time T (every quarter hour) the model sees only:
- actuals up to the quarter hour that ended one hour before T (measured lag: 22-50 min),
- the TSO forecasts published by T: intraday (A40) from 08:00 on the delivery day,
  otherwise day-ahead (A01, from 18:00 the day before),
- how wrong that TSO forecast was on the last known quarter hours,
- solar geometry and calendar,
- installed capacity as known two months earlier.

It predicts the error of the TSO forecast for target t = T + k quarter hours (the
residual idea Augur describes), in capacity-factor units. Walk-forward by month, scored
against three rules on exactly the same (issue time, target) pairs:
TSO forecast as published, TSO plus its last known error, and persistence.
"""
from __future__ import annotations

import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from . import baseline, features
from .config import UTC
from .model_da import capacity_known
from .store import Store
from .util import BERLIN, iso, month_start, months, next_month, now

LOG = logging.getLogger(__name__)

TECHS = ("solar", "wind_on", "wind_off")
LAG = baseline.KNOWN_AFTER_QH                     # last known quarter hour = T - 5 quarter hours
EVAL_K = baseline.HORIZONS_QH                     # 15 min .. 8 h, as in the baseline report
TRAIN_K = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
TRAIN_FROM = datetime(2024, 1, 16, tzinfo=UTC)
CALIBRATION_MONTHS = 2
PARAMS = dict(objective="l1", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
              seed=42, deterministic=True, force_row_wise=True, verbose=-1)
ROUNDS = 400
REPORT = "reports/model_id.json"
PREDICTIONS = "reports/model_id_backtest.parquet"
Q = pd.Timedelta(minutes=15)


def _at(s: pd.Series, ts: pd.DatetimeIndex) -> np.ndarray:
    return s.reindex(ts).to_numpy(dtype="float64")


def rows(y: pd.Series, da: pd.Series, id_: pd.Series, cap: pd.Series,
         issues: pd.DatetimeIndex, ks, a18: pd.Series | None = None) -> pd.DataFrame:
    """One row per (issue time, lead k). Every input is looked up at a time <= its availability."""
    out = []
    for k in ks:
        t = issues + k * Q
        L = issues - LAG * Q                      # last quarter hour known at T
        use_id, use_da = baseline._published(t, k)

        def tso(ts):                              # the TSO product used for this target, at other times
            return np.where(use_id, _at(id_, ts), np.where(use_da, _at(da, ts), np.nan))

        base = tso(t)
        c = _at(cap, t)
        err = {j: (_at(y, L - j * Q) - tso(L - j * Q)) / c for j in range(12)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-empty rows give NaN, as they should
            e_mean3h = np.nanmean(np.column_stack(list(err.values())), axis=1)
        tl = t.tz_convert(BERLIN)
        cz_t = features.cos_zenith(t + Q / 2, *features.DE_CENTER)
        cz_l = features.cos_zenith(L + Q / 2, *features.DE_CENTER)
        df = pd.DataFrame({
            "issue": issues, "target": t, "k": k,
            "base_cf": base / c, "base_is_id": use_id.astype("int8"),
            "da_cf": _at(da, t) / c,
            "e0": err[0], "e1": err[1], "e4": err[4], "e11": err[11], "e_mean3h": e_mean3h,
            "y_last_cf": _at(y, L) / c, "y_change_1h": (_at(y, L) - _at(y, L - 4 * Q)) / c,
            "tso_change": (base - tso(L)) / c,
            "hour": tl.hour + tl.minute / 60, "doy_sin": np.sin(2 * np.pi * tl.dayofyear / 365.25),
            "doy_cos": np.cos(2 * np.pi * tl.dayofyear / 365.25),
            "cz_t": cz_t, "cz_last": cz_l,
            "cs_ratio": features.clear_sky_ghi(cz_t) / np.maximum(features.clear_sky_ghi(cz_l), 50),
            # targets and rules, in MW
            "y": _at(y, t), "cap": c, "tso_best": base, "tso_last_error": base + err[0] * c,
            "persistence": _at(y, L),
            # benchmark only, never an input: the archived final version of the TSO's
            # continuously updated forecast, partly revised after delivery started
            "a18_final": _at(a18, t) if a18 is not None else np.nan,
        })
        out.append(df)
    return pd.concat(out, ignore_index=True)


FEATURES = ["k", "base_cf", "base_is_id", "da_cf", "e0", "e1", "e4", "e11", "e_mean3h", "y_last_cf",
            "y_change_1h", "tso_change", "hour", "doy_sin", "doy_cos", "cz_t", "cz_last", "cs_ratio"]


def fit(train: pd.DataFrame):
    import lightgbm as lgb      # only the model job needs it (requirements-model.txt)
    ok = train["y"].notna() & train["tso_best"].notna()
    target = (train.loc[ok, "y"] - train.loc[ok, "tso_best"]) / train.loc[ok, "cap"]
    return lgb.train(PARAMS, lgb.Dataset(train.loc[ok, FEATURES], target), num_boost_round=ROUNDS)


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """P50 in MW: TSO forecast plus the predicted correction."""
    return df["tso_best"].to_numpy() + model.predict(df[FEATURES]) * df["cap"].to_numpy()


def band(model, calib: pd.DataFrame) -> pd.DataFrame:
    """Per-lead P10/P90 offsets (capacity-factor units) from data the model did not train on."""
    res = (calib["y"] - predict(model, calib)) / calib["cap"]
    q = pd.DataFrame({"k": calib["k"], "res": res}).dropna().groupby("k")["res"].quantile([0.1, 0.9]).unstack()
    return q.rename(columns={0.1: "lo", 0.9: "hi"})


def backtest(store: Store, test_months: list[datetime]) -> pd.DataFrame:
    end = next_month(test_months[-1])
    idx = pd.date_range(TRAIN_FROM - pd.Timedelta(days=2), end + pd.Timedelta(hours=9), freq="15min")
    actual = baseline.load(store, "a75").reindex(idx)
    da = baseline.load(store, "a69_da").reindex(idx)
    id_ = baseline.load(store, "a69_id").reindex(idx)
    cap = capacity_known(store, idx)
    a18 = baseline.load(store, "a69_current").reindex(idx)
    hourly = pd.date_range(TRAIN_FROM, end, freq="h", inclusive="left")
    every_qh = pd.date_range(test_months[0] - pd.Timedelta(hours=8), end, freq="15min", inclusive="left")
    out = []
    for tech in TECHS:
        train_all = rows(actual[tech], da[tech], id_[tech], cap[tech], hourly, TRAIN_K)
        test_all = rows(actual[tech], da[tech], id_[tech], cap[tech], every_qh, EVAL_K, a18[tech])
        for m in test_months:
            m_ts = pd.Timestamp(m)
            calib_from = m_ts - pd.DateOffset(months=CALIBRATION_MONTHS)
            early = fit(train_all[train_all["target"] < calib_from])
            q = band(early, train_all[(train_all["target"] >= calib_from) & (train_all["target"] < m_ts)])
            model = fit(train_all[train_all["target"] < m_ts])
            test = test_all[(test_all["target"] >= m_ts) & (test_all["target"] < pd.Timestamp(next_month(m)))].copy()
            test["p50"] = predict(model, test)
            test["p10"] = test["p50"] + test["k"].map(q["lo"]).to_numpy() * test["cap"]
            test["p90"] = test["p50"] + test["k"].map(q["hi"]).to_numpy() * test["cap"]
            test["tech"] = tech
            out.append(test[["issue", "target", "k", "tech", "y", "p10", "p50", "p90",
                             "tso_best", "tso_last_error", "persistence", "a18_final"]])
            LOG.info("intraday %s %s done", tech, f"{m:%Y-%m}")
    res = pd.concat(out, ignore_index=True)
    res["tech"] = res["tech"].astype("string")
    return res


def paired_t(target: pd.Series, a: np.ndarray, b: np.ndarray) -> float:
    """t statistic of mean(|a| - |b|) with days as the unit (errors within a day are not
    independent) and the day-to-day autocorrelation discounted. Below -2: a is better."""
    day = pd.Series(np.abs(a) - np.abs(b), index=pd.DatetimeIndex(target)).groupby(lambda x: x.date()).mean()
    if len(day) < 5:
        return float("nan")
    if day.std() == 0:
        return 0.0
    rho = max(float(day.autocorr(1)), 0.0) if len(day) > 2 else 0.0
    n_eff = len(day) * (1 - rho) / (1 + rho)
    return float(day.mean() / (day.std() / np.sqrt(n_eff)))


def verdict(t: float) -> str:
    if np.isnan(t):
        return "too few days"
    return "model better" if t < -2 else ("benchmark better" if t > 2 else "not distinguishable")


def score(res: pd.DataFrame, cap_eval: pd.DataFrame) -> list[dict]:
    rows_ = []
    cols = ["p50", "tso_best", "tso_last_error", "persistence"]
    for (tech, k), g in res.groupby(["tech", "k"]):
        g = g.dropna(subset=["y", *cols])
        c = cap_eval[tech].reindex(pd.DatetimeIndex(g["target"])).to_numpy()
        row = {"tech": tech, "lead_min": int(15 * k), "n": int(len(g))}
        for name in cols:
            e = g[name] - g["y"]
            key = "model" if name == "p50" else name
            row[f"{key}_nmae_pct"] = round(float((e.abs() / c).mean() * 100), 2)
            row[f"{key}_rmse_mw"] = round(float(np.sqrt((e ** 2).mean())), 1)
        best_rule = min(("tso_best", "tso_last_error", "persistence"), key=lambda n: row[f"{n}_nmae_pct"])
        row["best_rule"] = best_rule
        row["skill_vs_best_rule"] = round(1 - row["model_nmae_pct"] / row[f"{best_rule}_nmae_pct"], 3)
        row["skill_vs_tso"] = round(1 - row["model_nmae_pct"] / row["tso_best_nmae_pct"], 3)
        t_rule = paired_t(g["target"], (g["p50"] - g["y"]).to_numpy() / c, (g[best_rule] - g["y"]).to_numpy() / c)
        row["t_vs_best_rule"] = round(t_rule, 1)
        row["verdict_vs_best_rule"] = verdict(t_rule)
        row["p10_p90_coverage"] = round(float(((g["y"] >= g["p10"]) & (g["y"] <= g["p90"])).mean()), 3)
        h = g.dropna(subset=["a18_final"])
        if len(h):
            ch = cap_eval[tech].reindex(pd.DatetimeIndex(h["target"])).to_numpy()
            row["a18_final_n"] = int(len(h))
            row["a18_final_nmae_pct"] = round(float(((h["a18_final"] - h["y"]).abs() / ch).mean() * 100), 2)
            row["model_on_a18_rows_nmae_pct"] = round(float(((h["p50"] - h["y"]).abs() / ch).mean() * 100), 2)
            row["skill_vs_a18_final"] = round(1 - row["model_on_a18_rows_nmae_pct"] / row["a18_final_nmae_pct"], 3)
            t_a18 = paired_t(h["target"], (h["p50"] - h["y"]).to_numpy() / ch, (h["a18_final"] - h["y"]).to_numpy() / ch)
            row["t_vs_a18_final"] = round(t_a18, 1)
            row["verdict_vs_a18_final"] = verdict(t_a18)
        rows_.append(row)
    return rows_


def run(store: Store, n_months: int = 12) -> int:
    t = now()
    end = month_start(t)
    test_months = months(datetime(end.year - 1, end.month, 1, tzinfo=UTC), end)[-n_months:]
    res = backtest(store, test_months)
    cap_eval = baseline.capacity_mw(store, pd.DatetimeIndex(res["target"].unique()).sort_values())
    report = {
        "generated_at": iso(t),
        "model": "v2: LightGBM on the error of the newest published TSO forecast",
        "issue_times": "every quarter hour; targets 15 min to 8 h ahead",
        "test_period": {"from": iso(test_months[0]), "to": iso(end)},
        "method": {
            "known_actuals": f"quarter hours that ended at least {15 * LAG - 15} min before issue time",
            "tso_input": "intraday forecast (A40) from 08:00 on the delivery day, else day-ahead (A01, 18:00 D-1)",
            "walk_forward": "each month predicted by a model trained only on earlier targets",
            "band": "P10/P90 per lead time from residuals on the two months before each test month",
            "compared_with": "TSO forecast as published, TSO plus its last known error, persistence; same rows",
            "significance": "paired test on daily mean absolute errors, autocorrelation-adjusted; "
                            "verdicts need |t| > 2",
            "a18_final": "archived final version of the TSO's continuously updated forecast (A18, also on "
                         "Energy-Charts as 'current'); keeps changing until ~30-80 min after delivery starts, "
                         "so it is stronger than what the TSO knew at issue time: beating it is conclusive, "
                         "losing to it is not. The exact A18 as of issue time is recorded since 2026-09-29",
            "limits": "history uses today's metered actuals, also as inputs; no weather input yet",
        },
        "scores": score(res, cap_eval),
    }
    store.put_json(REPORT, report)
    store.put_parquet(PREDICTIONS, res)
    store.commit(f"intraday model backtest {t:%Y-%m-%d}")
    for r in report["scores"]:
        LOG.info("%-8s %3d min  model %5.2f %%  TSO %5.2f %%  TSO+err %5.2f %%  A18 final %5s %%  skill vs best rule %+.1f %%  vs A18 final %s (%s)  cov %.0f %%",
                 r["tech"], r["lead_min"], r["model_nmae_pct"], r["tso_best_nmae_pct"],
                 r["tso_last_error_nmae_pct"], r.get("a18_final_nmae_pct"), 100 * r["skill_vs_best_rule"],
                 r.get("skill_vs_a18_final"), r.get("verdict_vs_a18_final"), 100 * r["p10_p90_coverage"])
    return 0
