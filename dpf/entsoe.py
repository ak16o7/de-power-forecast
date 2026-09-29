"""ENTSO-E Transparency Platform: A75 actual generation and A69 wind/solar forecasts."""
from __future__ import annotations

import io
import logging
import math
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timedelta

import pandas as pd
import requests

from .config import AREAS, ENTSOE_ENDPOINT, PSR, USER_AGENT, UTC

LOG = logging.getLogger(__name__)
COLUMNS = ["area", "psr", "ts", "mw"]


class NoData(Exception):
    """ENTSO-E answered 'No matching data found'."""


def api_key() -> str:
    key = os.environ.get("ENTSOE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ENTSOE_API_KEY is not set")
    return key


def mask(text: str) -> str:
    key = os.environ.get("ENTSOE_API_KEY", "")
    text = text.replace(key, "***") if key else text
    return re.sub(r"securityToken=[^&\s]+", "securityToken=***", text)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(elem: ET.Element, *names: str) -> str | None:
    wanted = {n.lower() for n in names}
    for node in elem.iter():
        if _local(node.tag).lower() in wanted and node.text:
            return node.text.strip()
    return None


def _duration(value: str | None) -> timedelta:
    m = re.fullmatch(r"PT(\d+)M", value or "") or re.fullmatch(r"PT(\d+)H", value or "")
    if not m:
        raise ValueError(f"unsupported resolution {value!r}")
    n = int(m.group(1))
    return timedelta(minutes=n) if value.endswith("M") else timedelta(hours=n)


def _xml_docs(content: bytes) -> list[bytes]:
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            return [zf.read(n) for n in zf.namelist() if n.lower().endswith(".xml")]
    return [content]


def parse(content: bytes) -> list[dict]:
    """Points of every TimeSeries as rows {psr, ts, mw, resolution}.

    Handles curveType A03 (omitted positions repeat the previous value) and skips
    consumption series (those carry outBiddingZone_Domain instead of inBiddingZone).
    """
    rows: list[dict] = []
    for doc in _xml_docs(content):
        root = ET.fromstring(doc)
        if _local(root.tag).lower() == "acknowledgement_marketdocument":
            reason = _child_text(root, "text") or "acknowledgement"
            if "no matching data" in reason.lower():
                raise NoData(reason)
            raise RuntimeError(f"ENTSO-E: {mask(reason)}")
        for ts in (n for n in root.iter() if _local(n.tag) == "TimeSeries"):
            if any(_local(n.tag) == "outBiddingZone_Domain.mRID" for n in ts.iter()):
                continue  # consumption (e.g. pumped storage load), not generation
            psr = _child_text(ts, "psrType")
            curve = _child_text(ts, "curveType")
            for period in (n for n in ts.iter() if _local(n.tag) == "Period"):
                start = pd.Timestamp(_child_text(period, "start")).tz_convert(UTC)
                end_txt = _child_text(period, "end")
                end = pd.Timestamp(end_txt).tz_convert(UTC) if end_txt else None
                res_txt = _child_text(period, "resolution")
                step = _duration(res_txt)
                pts = []
                for p in (n for n in period.iter() if _local(n.tag) == "Point"):
                    try:
                        pos, val = int(_child_text(p, "position")), float(_child_text(p, "quantity"))
                    except (TypeError, ValueError):
                        continue
                    if pos >= 1 and math.isfinite(val):
                        pts.append((pos, val))
                pts.sort()
                for i, (pos, val) in enumerate(pts):
                    repeat = 1
                    if curve == "A03":
                        if i + 1 < len(pts):
                            repeat = max(1, pts[i + 1][0] - pos)
                        elif end is not None:
                            repeat = max(1, int((end - (start + (pos - 1) * step)) / step))
                    for k in range(repeat):
                        t = start + (pos - 1 + k) * step
                        if end is not None and t >= end:
                            break
                        rows.append({"psr": psr, "ts": t, "mw": val, "resolution": res_txt})
    return rows


def to_quarter_hours(rows: list[dict]) -> pd.DataFrame:
    """Keep solar/wind rows, expand hourly/half-hourly points to 15-min, one value per (psr, ts)."""
    out = []
    for r in rows:
        name = PSR.get(r["psr"] or "")
        if not name:
            continue
        step = _duration(r["resolution"])
        n = max(1, int(step / timedelta(minutes=15)))
        for k in range(n):  # MW is a mean power: the same value holds for every quarter hour
            out.append((name, r["ts"] + timedelta(minutes=15 * k), r["mw"], n))
    df = pd.DataFrame(out, columns=["psr", "ts", "mw", "n"])
    if df.empty:
        return df[["psr", "ts", "mw"]]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    # Finer resolution wins if the same interval came twice (e.g. PT60M and PT15M series).
    df = df.sort_values("n", ascending=False, kind="stable")
    return df.groupby(["psr", "ts"], as_index=False)["mw"].last()


class Client:
    def __init__(self, session: requests.Session | None = None, min_interval: float = 0.2) -> None:
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", USER_AGENT)
        self.min_interval = min_interval  # ENTSO-E allows 400 requests/min per IP
        self._last = 0.0
        self.requests = 0

    def get(self, params: dict, start: datetime, end: datetime) -> bytes:
        payload = dict(params, securityToken=api_key(),
                       periodStart=start.astimezone(UTC).strftime("%Y%m%d%H%M"),
                       periodEnd=end.astimezone(UTC).strftime("%Y%m%d%H%M"))
        for attempt in range(5):
            wait = self.min_interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()
            self.requests += 1
            try:
                r = self.session.get(ENTSOE_ENDPOINT, params=payload, timeout=90)
            except requests.RequestException as exc:
                err = f"network: {type(exc).__name__}"
            else:
                if r.content[:2] != b"PK" and b"No matching data" in r.content[:4000]:
                    raise NoData("No matching data found")
                if r.status_code == 200:
                    return r.content
                if r.status_code in (400, 401, 403):
                    raise RuntimeError(mask(f"ENTSO-E {r.status_code}: {r.content[:300].decode('utf-8', 'ignore')}"))
                err = f"HTTP {r.status_code}"
            LOG.warning("ENTSO-E %s, retry %d", err, attempt + 1)
            time.sleep(min(60, 5 * 2 ** attempt))
        raise RuntimeError(f"ENTSO-E failed after retries ({err})")

    def actual(self, area: str, start: datetime, end: datetime) -> pd.DataFrame:
        """A75 actual generation per production type (all types, filtered to solar/wind)."""
        return self._frame(area, {"documentType": "A75", "processType": "A16",
                                  "in_Domain": AREAS[area]}, start, end)

    def forecast(self, area: str, process: str, start: datetime, end: datetime) -> pd.DataFrame:
        """A69 wind and solar forecast (process A01 day-ahead, A40 intraday, A18 current)."""
        return self._frame(area, {"documentType": "A69", "processType": process,
                                  "in_Domain": AREAS[area]}, start, end)

    def _frame(self, area, params, start, end) -> pd.DataFrame:
        try:
            df = to_quarter_hours(parse(self.get(params, start, end)))
        except NoData:
            df = pd.DataFrame(columns=["psr", "ts", "mw"])
        df.insert(0, "area", area)
        if not df.empty:
            df = df[(df["ts"] >= pd.Timestamp(start)) & (df["ts"] < pd.Timestamp(end))]
        return df.reset_index(drop=True)
