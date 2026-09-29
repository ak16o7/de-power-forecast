"""The honesty rule: every value carries the time it became available, and a
forecast issued at time T may only use values available at or before T.
"""
from __future__ import annotations

import pandas as pd


def latest(log: pd.DataFrame, keys: list[str], time_col: str = "seen_at") -> pd.DataFrame:
    """Most recent row per key."""
    if log is None or log.empty:
        return log
    return log.sort_values(time_col, kind="stable").groupby(keys, as_index=False, sort=False).last()


def changes(new: pd.DataFrame, known: pd.DataFrame | None, keys: list[str],
            value: str = "mw", tol: float = 0.05) -> pd.DataFrame:
    """Rows of `new` that are unseen or differ from the last known value by more than `tol`."""
    if known is None or known.empty:
        return new.reset_index(drop=True)
    m = new.merge(known[keys + [value]].rename(columns={value: "_old"}), on=keys, how="left")
    keep = m["_old"].isna() | ((m[value] - m["_old"]).abs() > tol)
    return new.loc[keep.to_numpy()].reset_index(drop=True)


def as_of(log: pd.DataFrame, when, keys: list[str], avail: str = "seen_at") -> pd.DataFrame:
    """State of the world at `when`: latest value per key among rows available <= when."""
    when = pd.Timestamp(when)
    return latest(log[log[avail] <= when], keys, avail)


def asof_join(targets: pd.DataFrame, log: pd.DataFrame, by: list[str],
              issue: str = "issue_time", avail: str = "seen_at") -> pd.DataFrame:
    """For every target row (by-keys + issue time) attach the value known at issue time.

    Rows whose value was not yet available at issue time get NaN, never a later value.
    """
    left = targets.sort_values(issue, kind="stable")
    right = log.sort_values(avail, kind="stable")
    out = pd.merge_asof(left, right, left_on=issue, right_on=avail, by=by,
                        direction="backward", allow_exact_matches=True)
    return out.sort_index()
