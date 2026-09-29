import pandas as pd

from dpf.store import LocalStore
from dpf.vintage import as_of, asof_join, changes, latest

K = ["area", "ts"]


def log():
    t = pd.Timestamp
    return pd.DataFrame({
        "area": ["DE", "DE", "DE"],
        "ts": [t("2026-09-29T10:00Z")] * 3,
        "mw": [100.0, 120.0, 125.0],
        "seen_at": [t("2026-09-29T10:20Z"), t("2026-09-29T10:50Z"), t("2026-09-29T13:00Z")],
    })


def test_changes_only_new_or_different():
    known = latest(log(), K)
    new = pd.DataFrame({"area": ["DE", "DE"], "ts": [pd.Timestamp("2026-09-29T10:00Z"), pd.Timestamp("2026-09-29T10:15Z")],
                        "mw": [125.02, 90.0]})
    out = changes(new, known, K)
    assert list(out["mw"]) == [90.0]          # 125.02 is within tolerance
    assert len(changes(new, None, K)) == 2


def test_as_of_never_sees_the_future():
    assert as_of(log(), "2026-09-29T10:10Z", K).empty
    assert as_of(log(), "2026-09-29T10:20Z", K)["mw"].item() == 100.0
    assert as_of(log(), "2026-09-29T12:00Z", K)["mw"].item() == 120.0


def test_asof_join_backtest_is_leak_free():
    targets = pd.DataFrame({"area": ["DE"] * 4, "ts": [pd.Timestamp("2026-09-29T10:00Z")] * 4,
                            "issue_time": pd.to_datetime(["2026-09-29T10:00Z", "2026-09-29T10:30Z",
                                                          "2026-09-29T11:00Z", "2026-09-29T14:00Z"])})
    out = asof_join(targets, log(), by=K)
    assert out["mw"].isna().iloc[0]
    assert list(out["mw"].iloc[1:]) == [100.0, 120.0, 125.0]
    assert (out["seen_at"].dropna() <= out["issue_time"][out["seen_at"].notna()]).all()


def test_local_store_staging(tmp_path):
    s = LocalStore(tmp_path)
    s.put("a/b/1.txt", b"x")
    s.put("a/b/2.txt", b"y")
    assert s.list("a") == ["a/b/1.txt", "a/b/2.txt"]
    s.commit("one")
    s.delete("a/b/")
    assert s.list("a") == []
    assert s.read("a/b/1.txt") == b"x"   # deletion is staged, the committed file is still readable
    s.commit("two")
    assert s.list("a") == [] and s.read("a/b/1.txt") is None
    assert s.commit("nothing") is False
