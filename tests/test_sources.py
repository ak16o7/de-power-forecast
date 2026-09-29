from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from dpf import entsoe, openmeteo
from dpf.config import MODELS, POINTS
from tests.fakes import NO_DATA, FakeOpenMeteo, gl_document

UTC = timezone.utc


# ---------------------------------------------------------------- ENTSO-E parsing
def test_parse_a01_curve_and_skip_consumption():
    xml = gl_document([
        {"psr": "B16", "start": "2026-09-29T10:00Z", "end": "2026-09-29T11:00Z",
         "points": [(1, 10), (2, 20), (3, 30), (4, 40)]},
        {"psr": "B16", "start": "2026-09-29T10:00Z", "end": "2026-09-29T11:00Z",
         "points": [(1, 999)], "consumption": True},
    ])
    df = entsoe.to_quarter_hours(entsoe.parse(xml))
    assert list(df["mw"]) == [10, 20, 30, 40]
    assert df["ts"].iloc[0] == pd.Timestamp("2026-09-29T10:00Z")
    assert set(df["psr"]) == {"solar"}


def test_parse_a03_repeats_omitted_positions():
    xml = gl_document([{"psr": "B19", "start": "2026-09-29T00:00Z", "end": "2026-09-29T01:00Z",
                        "curve": "A03", "points": [(1, 5), (3, 7)]}])
    df = entsoe.to_quarter_hours(entsoe.parse(xml))
    assert list(df["mw"]) == [5, 5, 7, 7]


def test_hourly_points_expand_to_quarter_hours_and_finer_wins():
    xml = gl_document([
        {"psr": "B18", "start": "2026-09-29T00:00Z", "end": "2026-09-29T01:00Z", "resolution": "PT60M",
         "points": [(1, 100)]},
        {"psr": "B18", "start": "2026-09-29T00:00Z", "end": "2026-09-29T00:15Z", "points": [(1, 90)]},
    ])
    df = entsoe.to_quarter_hours(entsoe.parse(xml)).set_index("ts")["mw"]
    assert len(df) == 4
    assert df.iloc[0] == 90 and list(df.iloc[1:]) == [100, 100, 100]


def test_no_data_and_secret_masking(monkeypatch):
    with pytest.raises(entsoe.NoData):
        entsoe.parse(NO_DATA)
    monkeypatch.setenv("ENTSOE_API_KEY", "abc-secret-123")
    assert "abc-secret-123" not in entsoe.mask("url?securityToken=abc-secret-123&x=1 and abc-secret-123")


# ---------------------------------------------------------------- Open-Meteo
def test_weight_matches_published_rule():
    assert openmeteo.weight(1, 10, 24 * 14) == 1
    assert openmeteo.weight(1, 15, 24 * 14) == 1.5
    assert openmeteo.weight(1, 15, 24 * 28) == 3
    assert openmeteo.weight(16, 20, 24 * 31) == pytest.approx(16 * 2 * 31 / 14)


def test_budget_stops_before_overspending():
    b = openmeteo.Budget(run_cap=20, day_left=100, minute_cap=1000)
    b.spend(16)
    with pytest.raises(openmeteo.BudgetExhausted):
        b.spend(16)
    b2 = openmeteo.Budget(run_cap=100, day_left=10, minute_cap=1000)
    with pytest.raises(openmeteo.BudgetExhausted):
        b2.spend(16)


def test_multi_location_frames_and_cell_selection():
    fake = FakeOpenMeteo()
    c = openmeteo.Client(session=fake)
    run = datetime(2026, 9, 29, 0, tzinfo=UTC)
    df = c.single_run(MODELS["icon_d2"], run, POINTS, 6)
    assert len(df) == len(POINTS) * 6
    assert set(df["point"]) == {p.name for p in POINTS}
    cells = sorted(p["cell_selection"] for _, p in fake.calls)
    assert cells == ["land", "sea"]  # onshore and offshore requested separately
    assert all(p["run"] == "2026-09-29T00:00" for _, p in fake.calls)
    assert c.weight_used == pytest.approx(len(POINTS))


def test_previous_runs_long_format():
    c = openmeteo.Client(session=FakeOpenMeteo())
    m = MODELS["ecmwf_ifs"]
    start = datetime(2026, 5, 1, tzinfo=UTC)
    df = c.previous_runs(m, POINTS[:2], start, start + timedelta(days=2), (1, 2))
    assert sorted(df["lead_days"].unique()) == [1, 2]
    assert len(df) == 2 * 2 * 48
    assert set(m.variables) <= set(df.columns)


def test_errors_are_classified():
    e = openmeteo.BadRequest("Cannot initialize WeatherVariable from invalid String value snow_depth_previous_day1 for key hourly")
    assert e.bad_variable == "snow_depth_previous_day1"
    assert openmeteo.RateLimited("Daily API request limit exceeded.").scope == "day"
    assert openmeteo.RateLimited("Minutely API request limit exceeded.").scope == "minute"
