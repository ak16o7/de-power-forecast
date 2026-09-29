"""Everything that defines *what* we collect lives here.

Changing a list below changes what the scheduled jobs fetch from the next run on.
Nothing here is secret: tokens come from the environment only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone

UTC = timezone.utc

HF_DATASET = os.environ.get("HF_DATASET", "akderekaan/de-power-forecast-data")
USER_AGENT = "de-power-forecast/0.1 (+https://github.com/ak16o7/de-power-forecast)"

# --------------------------------------------------------------------------- ENTSO-E
ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"

# Germany as a whole plus the four control areas (they add up to DE).
AREAS = {
    "DE": "10Y1001A1001A83F",
    "50HZ": "10YDE-VE-------2",
    "AMP": "10YDE-RWENET---I",
    "TTG": "10YDE-EON------1",
    "TBW": "10YDE-ENBW-----N",
}

# ENTSO-E production types -> our target names
PSR = {"B16": "solar", "B19": "wind_on", "B18": "wind_off"}

# A69 wind & solar forecasts: process type -> stream name
A69_PROCESSES = {
    "A01": "a69_da",       # day-ahead, published ~18:00 CET on D-1
    "A40": "a69_id",       # intraday, published ~08:00 CET on D
    "A18": "a69_current",  # "current", updated during the day; only the last version is archived
}

ENTSOE_HISTORY_START = datetime(2024, 1, 1, tzinfo=UTC)

# --------------------------------------------------------------------------- weather
@dataclass(frozen=True)
class Point:
    name: str
    lat: float
    lon: float
    kind: str  # "onshore" | "offshore"


POINTS: tuple[Point, ...] = (
    Point("SH", 54.3, 9.7, "onshore"),
    Point("NI_W", 52.9, 7.8, "onshore"),
    Point("NI_O", 52.6, 10.2, "onshore"),
    Point("MV", 53.8, 12.3, "onshore"),
    Point("BB", 52.4, 13.7, "onshore"),
    Point("ST", 52.0, 11.7, "onshore"),
    Point("NRW", 51.5, 7.5, "onshore"),
    Point("HE", 50.6, 9.0, "onshore"),
    Point("TH_SN", 51.0, 12.5, "onshore"),
    Point("RP_SL", 49.8, 7.3, "onshore"),
    Point("BW", 48.6, 9.0, "onshore"),
    Point("BY_N", 49.6, 11.2, "onshore"),
    Point("BY_S", 48.2, 12.0, "onshore"),
    Point("NS_BORKUM", 54.0, 6.5, "offshore"),
    Point("NS_HELGO", 54.5, 7.6, "offshore"),
    Point("OS_ARKONA", 54.8, 14.0, "offshore"),
)

_SOLAR = ["shortwave_radiation", "direct_radiation", "diffuse_radiation", "temperature_2m",
          "cloud_cover", "snow_depth"]
_COMMON_TAIL = ["wind_gusts_10m", "surface_pressure"]


@dataclass(frozen=True)
class WeatherModel:
    name: str              # Open-Meteo `models=` value
    meta_dir: str          # https://api.open-meteo.com/data/<meta_dir>/static/meta.json
    variables: tuple[str, ...]
    run_every_h: int       # model cycle
    horizon_h: int         # how far ahead we keep each run
    delay_h: float         # conservative "run init -> usable" delay for archived runs
    single_runs_from: datetime | None  # first run archived in the Single Runs API
    previous_runs_from: datetime | None
    previous_days: tuple[int, ...] = (1, 2)  # day1 = predicted >= 24 h before the valid time


# At most 10 variables per model: more than 10 doubles the API cost.
MODELS: dict[str, WeatherModel] = {
    "ecmwf_ifs": WeatherModel(
        "ecmwf_ifs", "ecmwf_ifs",
        tuple(_SOLAR + ["wind_speed_100m", "wind_direction_100m"] + _COMMON_TAIL),
        run_every_h=6, horizon_h=72, delay_h=8.5,   # measured 7.1-7.5 h
        single_runs_from=datetime(2024, 3, 14, tzinfo=UTC),
        previous_runs_from=datetime(2025, 10, 1, tzinfo=UTC),  # first backfill: empty before 2025-10
    ),
    "icon_d2": WeatherModel(
        "icon_d2", "dwd_icon_d2",
        tuple(_SOLAR + ["wind_speed_120m", "wind_direction_120m"] + _COMMON_TAIL),
        run_every_h=3, horizon_h=48, delay_h=2.5,   # measured 1.4-1.8 h
        single_runs_from=datetime(2026, 4, 2, tzinfo=UTC),
        previous_runs_from=datetime(2024, 1, 1, tzinfo=UTC),
        previous_days=(1,),                          # 48 h horizon: day2 would be empty
    ),
    "icon_eu": WeatherModel(
        "icon_eu", "dwd_icon_eu",
        tuple(_SOLAR + ["wind_speed_120m", "wind_direction_120m"] + _COMMON_TAIL),
        run_every_h=3, horizon_h=72, delay_h=4.5,   # measured 2.9-3.7 h
        single_runs_from=datetime(2026, 4, 2, tzinfo=UTC),
        previous_runs_from=datetime(2024, 1, 1, tzinfo=UTC),
    ),
}

# Open-Meteo free tier: 600/min, 5,000/h, 10,000/day, 300,000/month (fractional weights).
OM_DAILY_CAP = float(os.environ.get("OM_DAILY_CAP", 8000))
OM_RUN_CAP = float(os.environ.get("OM_RUN_CAP", 1500))       # per backfill job run
OM_MINUTE_CAP = float(os.environ.get("OM_MINUTE_CAP", 400))

# Order in which the weather backfill works through its queues.
BACKFILL_ORDER = (
    ("previous_runs", "ecmwf_ifs"),
    ("previous_runs", "icon_d2"),
    ("previous_runs", "icon_eu"),
    ("single_runs", "icon_d2"),
    ("single_runs", "ecmwf_ifs"),
    ("single_runs", "icon_eu"),
)

# --------------------------------------------------------------------------- capacity
ENERGY_CHARTS_CAPACITY = "https://api.energy-charts.info/installed_power"
