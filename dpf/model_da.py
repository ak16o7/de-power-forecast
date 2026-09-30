"""Day-ahead model v1: P10 / P50 / P90 for every quarter hour of day D, issued D-1 11:00.

- Inputs: ICON-EU forecasts made >= 48 h ahead (Open-Meteo previous runs, lead_days = 2)
  at 16 points, solar geometry, calendar. Nothing published after the issue time
  (checked in features.build).
- Target: capacity factor = MW / installed capacity. The capacity used is the value of
  two months earlier, because recent months are still revised when late registrations
  come in.
- Model: gradient boosted trees (LightGBM), one per technology and quantile, fixed
  settings (no tuning on the test period).
- P10/P90: quantile trees are too narrow out of sample, so the band is widened by a factor
  measured on the two months before M with a model that did not see them (conformal).
- Backtest: walk-forward by month. Month M is predicted by a model trained only on data
  before M, then scored against the TSO day-ahead forecast on the same quarter hours.
"""
from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from . import baseline, features
from .config import UTC
from .store import Store
from .util import iso, month_start, months, next_month, now

LOG = logging.getLogger(__name__)

TECHS = ("solar", "wind_on", "wind_off")
QUANTILES = (0.1, 0.5, 0.9)
TRAIN_FROM = datetime(2024, 3, 1, tzinfo=UTC)      # ICON-EU 48 h forecasts complete from here
CALIBRATION_MONTHS = 2                               # held out to size the P10-P90 band
PARAMS = dict(n_estimators=500, learning_rate=0.05, num_leaves=63, min_data_in_leaf=40,
              bagging_fraction=0.8, bagging_freq=1, feature_fraction=0.8, lambda_l2=1.0,
              seed=42, deterministic=True, force_row_wise=True, verbose=-1)   # same result every run
REPORT = "reports/model_da.json"
PREDICTIONS = "reports/model_da_backtest.parquet"


def capacity_known(store: Store, qh: pd.DatetimeIndex) -> pd.DataFrame:
    """Installed capacity (MW) as usable at issue time: the value two months back."""
    return baseline.capacity_mw(store, qh - pd.DateOffset(months=2)).set_axis(qh)


def fit_predict(X: pd.DataFrame, y: pd.Series, X_new: pd.DataFrame) -> pd.DataFrame:
    import lightgbm as lgb      # only the model job needs it (requirements-model.txt)
    ok = y.notna()
    params = {k: v for k, v in PARAMS.items() if k != "n_estimators"}
    data = lgb.Dataset(X[ok], y[ok], free_raw_data=False)
    out = {}
    for q in QUANTILES:
        booster = lgb.train({**params, "objective": "quantile", "alpha": q}, data,
                            num_boost_round=PARAMS["n_estimators"])
        out[f"p{int(q * 100)}"] = booster.predict(X_new)
    pred = pd.DataFrame(out, index=X_new.index).clip(lower=0, upper=1.05)
    pred[:] = np.sort(pred.to_numpy(), axis=1)          # no crossing quantiles
    return pred


def widening(pred: pd.DataFrame, actual: pd.Series, target: float = 0.8) -> float:
    """Factor k so that [p50 - k(p50-p10), p50 + k(p90-p50)] holds `target` of the actuals.

    Quantile trees are too sure of themselves out of sample; k is measured on data the
    model did not train on (conformal calibration) and then applied unchanged.
    """
    lo, mid, hi = pred["p10"], pred["p50"], pred["p90"]
    ok = actual.notna() & ((mid - lo) > 1e-4) & ((hi - mid) > 1e-4)
    if ok.sum() < 500:
        return 1.0
    need = np.maximum((mid - actual) / (mid - lo), (actual - mid) / (hi - mid))[ok]
    return float(max(1.0, np.quantile(need, target)))


def widen(pred: pd.DataFrame, k: float) -> pd.DataFrame:
    out = pred.copy()
    out["p10"] = (pred["p50"] - k * (pred["p50"] - pred["p10"])).clip(lower=0)
    out["p90"] = pred["p50"] + k * (pred["p90"] - pred["p50"])
    return out


def backtest(store: Store, test_months: list[datetime]) -> pd.DataFrame:
    end = next_month(test_months[-1])
    qh = pd.date_range(TRAIN_FROM, end, freq="15min", inclusive="left")
    feats = features.build(store, qh)
    actual = baseline.load(store, "a75").reindex(qh)
    tso = baseline.load(store, "a69_da").reindex(qh)
    cap = capacity_known(store, qh)
    rows = []
    for tech in TECHS:
        cols = features.columns_for(tech, feats)
        y = actual[tech] / cap[tech]
        for m in test_months:
            train = qh < m
            test = (qh >= m) & (qh < next_month(m))
            calib_from = m - pd.DateOffset(months=CALIBRATION_MONTHS)
            fit, calib = qh < calib_from, (qh >= calib_from) & train
            k = widening(fit_predict(feats.loc[fit, cols], y[fit], feats.loc[calib, cols]), y[calib])
            pred = widen(fit_predict(feats.loc[train, cols], y[train], feats.loc[test, cols]), k) \
                .mul(cap.loc[test, tech], axis=0)
            pred["interval_factor"] = k
            pred["tech"] = tech
            pred["actual"] = actual.loc[test, tech]
            pred["tso_da"] = tso.loc[test, tech]
            rows.append(pred)
            LOG.info("%s %s: trained on %d quarter hours, interval factor %.2f", tech, f"{m:%Y-%m}",
                     int(y[train].notna().sum()), k)
    out = pd.concat(rows).rename_axis("ts").reset_index()
    out["tech"] = out["tech"].astype("string")
    return out


def score(pred: pd.DataFrame, cap_eval: pd.DataFrame) -> list[dict]:
    """Model P50 vs TSO day-ahead on the quarter hours where both and the actual exist."""
    rows = []
    for tech, g in pred.groupby("tech"):
        g = g.dropna(subset=["p50", "actual", "tso_da"]).set_index("ts")
        c = cap_eval[tech].reindex(g.index)
        e_m, e_t = g["p50"] - g["actual"], g["tso_da"] - g["actual"]
        inside = ((g["actual"] >= g["p10"]) & (g["actual"] <= g["p90"])).mean()
        rows.append({
            "tech": tech, "n": int(len(g)),
            "model_mae_mw": round(float(e_m.abs().mean()), 1), "tso_mae_mw": round(float(e_t.abs().mean()), 1),
            "model_nmae_pct": round(float((e_m.abs() / c).mean() * 100), 2),
            "tso_nmae_pct": round(float((e_t.abs() / c).mean() * 100), 2),
            "model_bias_mw": round(float(e_m.mean()), 1),
            "skill_vs_tso": round(float(1 - e_m.abs().mean() / e_t.abs().mean()), 3),
            "p10_p90_coverage": round(float(inside), 3),
        })
    return rows


def monthly(pred: pd.DataFrame, cap_eval: pd.DataFrame) -> list[dict]:
    rows = []
    pred = pred.dropna(subset=["p50", "actual", "tso_da"])
    for (tech, m), g in pred.groupby(["tech", pred["ts"].dt.strftime("%Y-%m")]):
        c = cap_eval[tech].reindex(pd.DatetimeIndex(g["ts"])).to_numpy()
        rows.append({"month": m, "tech": tech,
                     "model_nmae_pct": round(float(((g["p50"] - g["actual"]).abs() / c).mean() * 100), 2),
                     "tso_nmae_pct": round(float(((g["tso_da"] - g["actual"]).abs() / c).mean() * 100), 2)})
    return rows


def run(store: Store, n_months: int = 12) -> int:
    t = now()
    end = month_start(t)
    test_months = months(datetime(end.year - 1, end.month, 1, tzinfo=UTC), end)[-n_months:]
    pred = backtest(store, test_months)
    cap_eval = baseline.capacity_mw(store, pd.DatetimeIndex(pred["ts"].unique()).sort_values())
    report = {
        "generated_at": iso(t),
        "model": "v1: LightGBM quantiles on ICON-EU forecasts made >= 48 h ahead (Open-Meteo previous runs)",
        "issue_time": "11:00 Europe/Berlin on the day before delivery",
        "test_period": {"from": iso(test_months[0]), "to": iso(end)},
        "method": {
            "walk_forward": "each month predicted by a model trained only on earlier data",
            "compared_with": "TSO day-ahead (A01), published 18:00 on D-1, i.e. 7 h later and with fresher weather",
            "normalisation": "nMAE = MAE / installed capacity of the month (same as the baseline report)",
            "capacity_input": "the model scales with the capacity known two months earlier",
            "interval": "P10-P90 widened by a factor fitted on the two months before each test month",
            "limits": "ICON-EU at >= 48 h lead is older than what is available at 11:00; "
                      "v1b will use the newest model runs once their archive is complete",
        },
        "scores": score(pred, cap_eval),
        "monthly": monthly(pred, cap_eval),
    }
    store.put_json(REPORT, report)
    store.put_parquet(PREDICTIONS, pred)
    store.commit(f"day-ahead model backtest {t:%Y-%m-%d}")
    for r in report["scores"]:
        LOG.info("%-8s model nMAE %5.2f %%  TSO %5.2f %%  skill %+.1f %%  P10-P90 coverage %.0f %%",
                 r["tech"], r["model_nmae_pct"], r["tso_nmae_pct"], 100 * r["skill_vs_tso"],
                 100 * r["p10_p90_coverage"])
    return 0
