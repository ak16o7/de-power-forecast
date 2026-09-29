from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from .config import UTC

BERLIN = ZoneInfo("Europe/Berlin")


def now() -> datetime:
    """UTC now; DPF_NOW=2026-09-29T12:00Z pins it for tests."""
    fixed = os.environ.get("DPF_NOW")
    if fixed:
        return datetime.fromisoformat(fixed.replace("Z", "+00:00")).astimezone(UTC)
    return datetime.now(UTC)


def floor(t: datetime, minutes: int) -> datetime:
    t = t.astimezone(UTC).replace(second=0, microsecond=0)
    return t - timedelta(minutes=t.minute % minutes) if minutes < 60 else \
        t.replace(minute=0) - timedelta(hours=t.hour % (minutes // 60))


def berlin_midnight(t: datetime, add_days: int = 0) -> datetime:
    d = t.astimezone(BERLIN).date() + timedelta(days=add_days)
    return datetime(d.year, d.month, d.day, tzinfo=BERLIN).astimezone(UTC)


def month_start(t: datetime) -> datetime:
    return datetime(t.year, t.month, 1, tzinfo=UTC)


def next_month(t: datetime) -> datetime:
    return datetime(t.year + (t.month == 12), t.month % 12 + 1, 1, tzinfo=UTC)


def months(start: datetime, end: datetime) -> list[datetime]:
    """UTC month starts m with m < end, beginning at the month of `start`."""
    out, m = [], month_start(start)
    while m < end:
        out.append(m)
        m = next_month(m)
    return out


def iso(t: datetime | None) -> str | None:
    return None if t is None else t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str | None) -> datetime | None:
    return None if not s else datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(UTC)


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("urllib3", "httpx", "huggingface_hub.file_download", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
