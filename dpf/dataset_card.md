---
license: cc-by-4.0
pretty_name: DE Power Forecast data
language:
- en
- de
tags:
- energy
- electricity
- renewable-energy
- time-series
- forecasting
- germany
size_categories:
- 1M<n<10M
configs:
- config_name: actual_generation
  data_files: "entsoe/a75/*/*/*.parquet"
- config_name: tso_forecast_day_ahead
  data_files: "entsoe/a69_da/*/*/*.parquet"
- config_name: tso_forecast_intraday
  data_files: "entsoe/a69_id/*/*/*.parquet"
- config_name: vintages
  data_files: "vintages/entsoe/*/*.parquet"
- config_name: weather_runs_ecmwf_ifs
  data_files: "weather/runs/ecmwf_ifs/*/*.parquet"
- config_name: weather_runs_icon_d2
  data_files: "weather/runs/icon_d2/*/*.parquet"
- config_name: weather_runs_icon_eu
  data_files: "weather/runs/icon_eu/*/*.parquet"
- config_name: weather_previous_runs_ecmwf_ifs
  data_files: "weather/previous_runs/ecmwf_ifs/*/*.parquet"
- config_name: weather_previous_runs_icon_d2
  data_files: "weather/previous_runs/icon_d2/*/*.parquet"
- config_name: weather_previous_runs_icon_eu
  data_files: "weather/previous_runs/icon_eu/*/*.parquet"
- config_name: installed_capacity
  data_files: "capacity/latest.parquet"
---

# DE Power Forecast – data

Input and target data for forecasting German solar, onshore wind and offshore wind
generation, collected automatically by the GitHub Actions jobs in
[ak16o7/de-power-forecast](https://github.com/ak16o7/de-power-forecast).

**The rule behind the layout:** every value carries the time it became available.
A forecast issued at time T may only use rows with `seen_at` / `available_at` ≤ T.
That is what makes backtests on this data honest.

## Files

| Path | What | Time columns |
|---|---|---|
| `entsoe/a75/{area}/{YYYY}/{YYYY-MM}.parquet` | Actual generation (ENTSO-E A75), 15 min, MW, latest known value | `ts` (interval start, UTC), `fetched_at` |
| `entsoe/a69_da/…`, `entsoe/a69_id/…` | TSO wind & solar forecasts, day-ahead (A01, published ~18:00 CET D-1) and intraday (A40, ~08:00 CET D) | `ts`, `fetched_at` |
| `vintages/entsoe/{YYYY}/{YYYY-MM-DD}.parquet` | Recorder log every 15 min: each new or changed value of A75 and A69 (day-ahead, intraday, current) with the time we saw it | `ts`, `seen_at` |
| `vintages/entsoe/parts/{day}/…` | Same for the current UTC day, compacted after midnight | |
| `weather/runs/{model}/{YYYY-MM}/{run}.parquet` | One file per weather model run, 16 points, hourly | `run`, `valid`, `available_at`, `fetched_at`, `source` (`live` = recorded when published, `archive` = backfilled) |
| `weather/previous_runs/{model}/{YYYY}/{YYYY-MM}.parquet` | Values predicted ≥ 24 h (`lead_days`=1) and ≥ 48 h (`lead_days`=2) before `valid` | `valid`, `available_at` = valid − N·24 h + model delay (upper bound) |
| `capacity/latest.parquet`, `capacity/snapshots/{date}.parquet` | Installed capacity per month (GW), snapshots because it is revised later | `month`, `seen_at` |
| `state/*.json`, `status.json` | Job bookkeeping and health | |

Areas: `DE` (Germany) and the control areas `50HZ`, `AMP` (Amprion), `TTG` (TenneT), `TBW` (TransnetBW).
Targets: `solar`, `wind_on`, `wind_off`.

Weather models: `ecmwf_ifs` (ECMWF IFS HRES 9 km), `icon_d2` (DWD ICON-D2 2 km), `icon_eu` (DWD ICON-EU 7 km).
Points: 13 onshore regions and 3 offshore sites (North Sea ×2, Baltic Sea), see `dpf/config.py` in the code repo.

## Sources and licences

- ENTSO-E Transparency Platform (A75, A69) – data item on ENTSO-E's list of data for free re-use, CC BY 4.0. Source: ENTSO-E Transparency Platform.
- Weather: [Open-Meteo](https://open-meteo.com/) (CC BY 4.0), underlying models by ECMWF and Deutscher Wetterdienst.
- Installed capacity: [Energy-Charts](https://www.energy-charts.info/) (Fraunhofer ISE), CC BY 4.0.

This dataset is published under CC BY 4.0; please credit the sources above.
Values are provided as received, without warranty. Actuals are provisional until the TSOs publish metered values.
