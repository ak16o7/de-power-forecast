"""python -m dpf record | daily | weather-backfill | report | backtest-da | backtest-id"""
from __future__ import annotations

import argparse
import sys

from .store import open_store
from .util import setup_logging


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="dpf")
    ap.add_argument("job", choices=["record", "daily", "weather-backfill", "report", "backtest-da", "backtest-id"])
    ap.add_argument("--max-seconds", type=float, default=None, help="time budget for backfill jobs")
    args = ap.parse_args(argv)
    setup_logging()
    store = open_store()
    if args.job == "record":
        from .record import run
        return run(store)
    if args.job == "backtest-id":
        from .model_id import run
        return run(store)
    if args.job == "backtest-da":
        from .model_da import run
        return run(store)
    if args.job == "report":
        from .baseline import run
        return run(store)
    if args.job == "daily":
        from .daily import run
        return run(store, **({"budget_s": args.max_seconds} if args.max_seconds else {}))
    from .weather_backfill import run
    return run(store, **({"max_s": args.max_seconds} if args.max_seconds else {}))


if __name__ == "__main__":
    sys.exit(main())
