"""Offline stand-ins for ENTSO-E and Open-Meteo (no network in tests)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

UTC = timezone.utc
NS = "urn:iec62325.351:tc57wg16:451-6:generationloaddocument:3:0"


def gl_document(series: list[dict]) -> bytes:
    """series: {psr, start, end, resolution, points: [(pos, qty)], curve?, consumption?}"""
    ts_xml = []
    for i, s in enumerate(series, 1):
        dom = ("<outBiddingZone_Domain.mRID codingScheme=\"A01\">X</outBiddingZone_Domain.mRID>"
               if s.get("consumption") else
               "<inBiddingZone_Domain.mRID codingScheme=\"A01\">X</inBiddingZone_Domain.mRID>")
        pts = "".join(f"<Point><position>{p}</position><quantity>{q}</quantity></Point>" for p, q in s["points"])
        ts_xml.append(
            f"<TimeSeries><mRID>{i}</mRID><businessType>A01</businessType>{dom}"
            f"<quantity_Measure_Unit.name>MAW</quantity_Measure_Unit.name>"
            f"<curveType>{s.get('curve', 'A01')}</curveType>"
            f"<MktPSRType><psrType>{s['psr']}</psrType></MktPSRType>"
            f"<Period><timeInterval><start>{s['start']}</start><end>{s['end']}</end></timeInterval>"
            f"<resolution>{s.get('resolution', 'PT15M')}</resolution>{pts}</Period></TimeSeries>")
    return (f'<?xml version="1.0" encoding="UTF-8"?><GL_MarketDocument xmlns="{NS}">'
            f"<mRID>x</mRID><createdDateTime>2026-09-29T12:00:00Z</createdDateTime>"
            + "".join(ts_xml) + "</GL_MarketDocument>").encode()


NO_DATA = (b'<?xml version="1.0" encoding="UTF-8"?><Acknowledgement_MarketDocument '
           b'xmlns="urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"><Reason><code>999</code>'
           b"<text>No matching data found for Data item ...</text></Reason></Acknowledgement_MarketDocument>")


class Resp:
    def __init__(self, status: int, content: bytes = b"", data=None) -> None:
        self.status_code = status
        self.content = content if data is None else json.dumps(data).encode()
        self.text = self.content.decode("utf-8", "ignore")

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _t(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H%M").replace(tzinfo=UTC)


class FakeEntsoe:
    """Deterministic MW values; `bump` shifts values to simulate revisions."""

    def __init__(self) -> None:
        self.headers: dict = {}
        self.calls: list[dict] = []
        self.bump: dict[str, float] = {}      # stream -> added MW
        self.no_data_areas: set[str] = set()
        self.fail: set[tuple[str, str]] = set()  # (documentType/processType, area code)

    @staticmethod
    def value(psr: str, t: datetime, area: str) -> float:
        base = {"B16": 1000, "B18": 300, "B19": 2000}[psr]
        return round(base + (t.hour * 4 + t.minute // 15) * 1.25 + len(area), 2)

    def get(self, url, params=None, timeout=None):
        p = dict(params)
        self.calls.append(p)
        start, end = _t(p["periodStart"]), _t(p["periodEnd"])
        kind = p["documentType"] if p["documentType"] == "A75" else f"A69/{p['processType']}"
        if (kind, p["in_Domain"]) in self.fail:
            return Resp(503, b"unavailable")
        if p["in_Domain"] in self.no_data_areas:
            return Resp(200, NO_DATA)
        stream = "a75" if kind == "A75" else {"A69/A01": "a69_da", "A69/A40": "a69_id", "A69/A18": "a69_current"}[kind]
        n = int((end - start) / timedelta(minutes=15))
        series = []
        for psr in ("B16", "B18", "B19", "B04"):
            if psr == "B04" and kind != "A75":
                continue
            pts = [(i + 1, (self.value(psr, start + i * timedelta(minutes=15), p["in_Domain"]) if psr != "B04" else 5000)
                    + self.bump.get(stream, 0)) for i in range(n)]
            series.append({"psr": psr, "start": start.strftime("%Y-%m-%dT%H:%MZ"),
                           "end": end.strftime("%Y-%m-%dT%H:%MZ"), "points": pts})
        return Resp(200, gl_document(series))


class FakeOpenMeteo:
    """Open-Meteo endpoints; values depend on (valid time, variable, location)."""

    def __init__(self, meta_runs: dict[str, datetime] | None = None) -> None:
        self.headers: dict = {}
        self.calls: list[tuple[str, dict]] = []
        self.meta_runs = meta_runs or {}
        self.reject_vars: set[str] = set()
        self.empty_before: datetime | None = None     # archive start
        self.rate_limit_after: int | None = None       # 429 after N data calls

    def get(self, url, params=None, timeout=None):
        if url.endswith("meta.json"):
            d = url.split("/data/")[1].split("/")[0]
            run = self.meta_runs.get(d)
            if run is None:
                return Resp(404, b"")
            return Resp(200, data={"last_run_initialisation_time": int(run.timestamp()),
                                   "last_run_availability_time": int(run.timestamp()) + 5400})
        p = dict(params)
        self.calls.append((url, p))
        if self.rate_limit_after is not None and len(self.calls) > self.rate_limit_after:
            return Resp(429, data={"error": True, "reason": "Hourly API request limit exceeded. Please try again in the next hour."})
        hourly = p["hourly"].split(",")
        for v in hourly:
            if v in self.reject_vars or any(v.startswith(r + "_previous") for r in self.reject_vars):
                return Resp(400, data={"error": True, "reason": f"Cannot initialize WeatherVariable from invalid String value {v} for key hourly"})
        start = datetime.fromisoformat(p["start_hour"]).replace(tzinfo=UTC)
        end = datetime.fromisoformat(p["end_hour"]).replace(tzinfo=UTC)
        times = pd.date_range(start, end, freq="h")
        lats = p["latitude"].split(",")
        empty = self.empty_before is not None and end < self.empty_before
        if "run" in p:
            run = datetime.fromisoformat(p["run"]).replace(tzinfo=UTC)
            empty = empty or (self.empty_before is not None and run < self.empty_before)
        out = []
        for i, lat in enumerate(lats):
            h = {"time": [int(t.timestamp()) for t in times]}
            for v in hourly:
                h[v] = [None if empty else round(float(lat) + t.hour + len(v) * 0.1, 2) for t in times]
            out.append({"latitude": float(lat), "hourly": h})
        return Resp(200, data=out if len(lats) > 1 else out[0])
