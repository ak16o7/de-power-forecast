"""Model inputs for every target quarter hour, built only from what was known at issue time.

Day-ahead issue time: 11:00 Europe/Berlin on the day before delivery (the EPEX day-ahead
auction closes at 12:00). Weather comes from Open-Meteo previous runs with lead_days = 2:
every value was predicted at least 48 h before its valid time and counts as available
48 h - model delay before it, which is before 11:00 on D-1 for every hour of day D.
`build()` checks that for every row it uses and refuses to continue otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODELS, POINTS, UTC
from .store import Store
from .util import BERLIN

ISSUE_HOUR = 11                     # local time on D-1
RADIATION = ("shortwave_radiation", "direct_radiation", "diffuse_radiation")  # means of the preceding hour
DE_CENTER = (51.2, 10.4)


def issue_time(target: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """D-1 11:00 local for every target quarter hour (DST-safe)."""
    local_day = target.tz_convert(BERLIN).tz_localize(None).normalize()   # calendar day, no offsets
    issue = local_day - pd.Timedelta(days=1) + pd.Timedelta(hours=ISSUE_HOUR)
    return issue.tz_localize(BERLIN, ambiguous="NaT", nonexistent="shift_forward").tz_convert(UTC)


def cos_zenith(times: pd.DatetimeIndex, lat: float, lon: float) -> np.ndarray:
    """Cosine of the solar zenith angle (NOAA approximation, good to a few tenths of a degree)."""
    t = times.tz_convert(UTC)
    hour = t.hour + t.minute / 60
    g = 2 * np.pi / 365 * (t.dayofyear - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                       - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
            + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hour * 60 + eqtime + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    return np.asarray(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha))


def clear_sky_ghi(cz: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky global horizontal irradiance, W/m2."""
    cz = np.clip(cz, 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cz > 0.01, 1098 * cz * np.exp(-0.057 / np.maximum(cz, 0.01)), 0.0)


def load_weather(store: Store, model: str, lead_days: int = 2) -> pd.DataFrame:
    frames = [store.read_parquet(p) for p in store.list(f"weather/previous_runs/{model}")]
    df = pd.concat([f for f in frames if f is not None], ignore_index=True)
    return df[df["lead_days"] == lead_days].reset_index(drop=True)


def to_quarter_hours(weather: pd.DataFrame, qh: pd.DatetimeIndex) -> pd.DataFrame:
    """Wide frame (column = point__variable) on quarter-hour centres.

    Radiation is an average of the preceding hour, so it is placed at the middle of that
    hour before interpolating; everything else is an instantaneous value.
    """
    centres = qh + pd.Timedelta(minutes=7.5)
    variables = [c for c in weather.columns if c not in ("point", "valid", "lead_days", "available_at", "fetched_at")]
    wide = weather.pivot_table(index="valid", columns="point", values=variables, aggfunc="last")
    out = {}
    # one time unit everywhere: pandas may hand out us or ns depending on how a stamp was made
    hours = pd.DatetimeIndex(wide.index).as_unit("us")
    x = centres.as_unit("us").asi8
    max_gap = 3 * 3600 * 10**6
    for var in wide.columns.get_level_values(0).unique():   # all-empty variables drop out here
        shift = 30 * 60 * 10**6 if var in RADIATION else 0
        xp_all = hours.asi8 - shift
        for point in wide[var].columns:
            fp = wide[(var, point)].to_numpy(dtype="float64")
            ok = ~np.isnan(fp)
            if ok.sum() < 2:
                continue
            xp, yp = xp_all[ok], fp[ok]
            vals = np.interp(x, xp, yp, left=np.nan, right=np.nan)
            # never bridge a hole in the weather data longer than 3 h
            right = np.clip(np.searchsorted(xp, x), 1, len(xp) - 1)
            span = xp[right] - xp[right - 1]
            out[f"{point}__{var}"] = np.where(span <= max_gap, vals, np.nan)
    return pd.DataFrame(out, index=qh)


def _epoch(t) -> np.ndarray:
    """Microseconds since 1970 as float (NaT -> NaN), for rolling maxima and comparisons."""
    idx = pd.DatetimeIndex(t).tz_convert(UTC).as_unit("us")
    out = idx.asi8.astype("float64")
    out[idx.isna()] = np.nan
    return out


def availability_check(weather: pd.DataFrame, qh: pd.DatetimeIndex) -> None:
    """Every weather value that can reach a target (valid within 2 h of it, the interpolation
    reach) must have been available at that target's issue time. Raises otherwise."""
    avail = weather.groupby("valid")["available_at"].max().sort_index()
    avail = pd.Series(_epoch(avail), index=avail.index)
    reach = avail.rolling("4h").max()                       # max over (v - 4 h, v]
    lookup = reach.reindex(qh.floor("h") + pd.Timedelta(hours=2), method="ffill").to_numpy()
    issue = _epoch(issue_time(qh))
    bad = ~np.isnan(lookup) & (lookup > issue)
    if bad.any():
        raise RuntimeError(f"{int(bad.sum())} targets would use weather published after their issue time")


def build(store: Store, qh: pd.DatetimeIndex, model: str = "icon_eu") -> pd.DataFrame:
    weather = load_weather(store, model)
    weather = weather[(weather["valid"] >= qh.min() - pd.Timedelta(hours=3))
                      & (weather["valid"] <= qh.max() + pd.Timedelta(hours=3))]
    availability_check(weather, qh)
    feats = to_quarter_hours(weather, qh)
    centres = qh + pd.Timedelta(minutes=7.5)
    local = centres.tz_convert(BERLIN)
    feats["hour_local"] = local.hour + local.minute / 60
    feats["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    feats["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    feats["cos_zenith"] = cos_zenith(centres, *DE_CENTER)
    feats["ghi_clear"] = clear_sky_ghi(feats["cos_zenith"].to_numpy())
    wind_var = next(v for v in MODELS[model].variables if v.startswith("wind_speed_"))
    dir_var = next(v for v in MODELS[model].variables if v.startswith("wind_direction_"))
    for p in POINTS:
        cs = clear_sky_ghi(cos_zenith(centres, p.lat, p.lon))
        ghi = feats.get(f"{p.name}__shortwave_radiation")
        if ghi is not None:
            feats[f"{p.name}__clear_index"] = np.where(cs > 50, ghi / np.maximum(cs, 1), np.nan)
        d = feats.get(f"{p.name}__{dir_var}")
        if d is not None:
            feats[f"{p.name}__dir_sin"] = np.sin(np.radians(d))
            feats[f"{p.name}__dir_cos"] = np.cos(np.radians(d))
            feats = feats.drop(columns=[f"{p.name}__{dir_var}"])
    onshore = [p.name for p in POINTS if p.kind == "onshore"]
    offshore = [p.name for p in POINTS if p.kind == "offshore"]
    ws_on = feats[[f"{n}__{wind_var}" for n in onshore if f"{n}__{wind_var}" in feats]]
    feats["onshore_ws_mean"] = ws_on.mean(axis=1)
    feats["onshore_ws3_mean"] = (ws_on.clip(upper=90) ** 3).mean(axis=1)
    ws_off = feats[[f"{n}__{wind_var}" for n in offshore if f"{n}__{wind_var}" in feats]]
    feats["offshore_ws_mean"] = ws_off.mean(axis=1)
    ghi_cols = [f"{n}__shortwave_radiation" for n in onshore if f"{n}__shortwave_radiation" in feats]
    feats["onshore_ghi_mean"] = feats[ghi_cols].mean(axis=1)
    return feats


def columns_for(tech: str, feats: pd.DataFrame) -> list[str]:
    """Feature subset per technology (solar: radiation and clouds, wind: wind and air)."""
    common = ["hour_local", "doy_sin", "doy_cos", "cos_zenith", "ghi_clear"]
    onshore = {p.name for p in POINTS if p.kind == "onshore"}
    offshore = {p.name for p in POINTS if p.kind == "offshore"}
    solar_vars = ("shortwave_radiation", "direct_radiation", "diffuse_radiation", "cloud_cover",
                  "temperature_2m", "snow_depth", "clear_index")
    wind_vars = ("wind_speed_", "wind_gusts_10m", "dir_sin", "dir_cos", "temperature_2m", "surface_pressure")

    def pick(points, vars_):
        return [c for c in feats.columns if "__" in c and c.split("__")[0] in points
                and any(c.split("__")[1].startswith(v) for v in vars_)]
    if tech == "solar":
        return common + pick(onshore, solar_vars) + ["onshore_ghi_mean"]
    if tech == "wind_on":
        return common + pick(onshore, wind_vars) + ["onshore_ws_mean", "onshore_ws3_mean"]
    coast = offshore | {"SH", "NI_W"}
    return common + pick(coast, wind_vars) + ["offshore_ws_mean"]
