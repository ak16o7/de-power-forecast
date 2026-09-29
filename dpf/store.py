"""Where the data lives: the public Hugging Face dataset, or a local folder for tests.

Jobs stage writes and commit them in batches. Every job owns its own paths, so
the recorder, the ENTSO-E job and the weather backfill can commit at the same
time without touching each other's files.
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

LOG = logging.getLogger(__name__)


class Store:
    def __init__(self) -> None:
        self._staged: dict[str, bytes | None] = {}  # path -> bytes, None = delete

    # -- implemented by backends
    def _read(self, path: str) -> bytes | None: raise NotImplementedError
    def _list(self, prefix: str) -> list[str]: raise NotImplementedError
    def _commit(self, adds: dict[str, bytes], deletes: list[str], message: str) -> None: raise NotImplementedError

    # -- public API
    def read(self, path: str) -> bytes | None:
        if path in self._staged:
            return self._staged[path]
        return self._read(path)

    def list(self, prefix: str) -> list[str]:
        prefix = prefix.rstrip("/")
        found = set(self._list(prefix))
        for p, data in self._staged.items():
            if p.startswith(prefix + "/"):
                if data is None:
                    found.discard(p)
                else:
                    found.add(p)
        deleted_dirs = [p for p, d in self._staged.items() if d is None]
        return sorted(p for p in found if not any(p.startswith(d.rstrip("/") + "/") for d in deleted_dirs))

    def put(self, path: str, data: bytes) -> None:
        self._staged[path] = data

    def delete(self, path: str) -> None:
        self._staged[path] = None

    def squash_history(self) -> bool:
        """Drop old file versions (the data carries its own timestamps). No-op locally."""
        return False

    @property
    def pending(self) -> int:
        return len(self._staged)

    def commit(self, message: str) -> bool:
        if not self._staged:
            return False
        adds = {p: d for p, d in self._staged.items() if d is not None}
        deletes = [p for p, d in self._staged.items() if d is None]
        self._commit(adds, deletes, message)
        self._staged.clear()
        return True

    # -- helpers
    def read_parquet(self, path: str) -> pd.DataFrame | None:
        data = self.read(path)
        return None if data is None else pd.read_parquet(io.BytesIO(data))

    def put_parquet(self, path: str, df: pd.DataFrame) -> None:
        df = df.copy()
        for c in df.columns:  # one timestamp type across all files, whatever pandas inferred
            if isinstance(df[c].dtype, pd.DatetimeTZDtype):
                df[c] = df[c].dt.tz_convert("UTC").astype("datetime64[us, UTC]")
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, compression="zstd")
        self.put(path, buf.getvalue())

    def read_json(self, path: str, default=None):
        data = self.read(path)
        return default if data is None else json.loads(data)

    def put_json(self, path: str, obj) -> None:
        self.put(path, json.dumps(obj, indent=1, sort_keys=True, default=str).encode())


class LocalStore(Store):
    def __init__(self, root: str | Path) -> None:
        super().__init__()
        self.root = Path(root)

    def _read(self, path: str) -> bytes | None:
        f = self.root / path
        return f.read_bytes() if f.is_file() else None

    def _list(self, prefix: str) -> list[str]:
        base = self.root / prefix
        if not base.exists():
            return []
        return [p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file()]

    def _commit(self, adds, deletes, message) -> None:
        import shutil
        for p in deletes:
            f = self.root / p
            if f.is_dir():
                shutil.rmtree(f)
            elif f.exists():
                f.unlink()
        for p, data in adds.items():
            f = self.root / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        LOG.info("local commit: %s (+%d -%d)", message, len(adds), len(deletes))


class HFStore(Store):
    """Public HF dataset. Reads work without a token; writes need HF_TOKEN."""

    def __init__(self, repo_id: str, token: str | None = None) -> None:
        super().__init__()
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi(token=token)

    def _read(self, path: str) -> bytes | None:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError
        for attempt in range(4):
            try:
                local = hf_hub_download(self.repo_id, path, repo_type="dataset", token=self.token)
                return Path(local).read_bytes()
            except (EntryNotFoundError, RemoteEntryNotFoundError):
                return None
            except Exception as exc:  # network hiccup: retry, then give up loudly
                if attempt == 3:
                    raise
                LOG.warning("HF read %s failed (%s), retrying", path, type(exc).__name__)
                time.sleep(2 * (attempt + 1))
        return None

    def _list(self, prefix: str) -> list[str]:
        from huggingface_hub.errors import EntryNotFoundError, RemoteEntryNotFoundError
        try:
            items = self.api.list_repo_tree(self.repo_id, path_in_repo=prefix or None, recursive=True,
                                            repo_type="dataset")
            return [i.path for i in items if i.__class__.__name__ == "RepoFile"]
        except (EntryNotFoundError, RemoteEntryNotFoundError):
            return []
        except Exception as exc:
            if "404" in str(exc):
                return []
            raise

    def squash_history(self) -> bool:
        self.api.super_squash_history(self.repo_id, repo_type="dataset",
                                      commit_message="Squash history (monthly housekeeping)")
        return True

    def _commit(self, adds, deletes, message) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        ops = [CommitOperationDelete(path_in_repo=p, is_folder="auto") for p in deletes]
        ops += [CommitOperationAdd(path_in_repo=p, path_or_fileobj=d) for p, d in adds.items()]
        for attempt in range(5):
            try:
                self.api.create_commit(self.repo_id, ops, commit_message=message, repo_type="dataset")
                LOG.info("HF commit: %s (+%d -%d)", message, len(adds), len(deletes))
                return
            except Exception as exc:
                if attempt == 4:
                    raise
                wait = 5 * (attempt + 1) ** 2
                LOG.warning("HF commit failed (%s: %s), retry in %ss", type(exc).__name__, str(exc)[:200], wait)
                time.sleep(wait)


def open_store() -> Store:
    """DPF_STORE=local:/path for tests and dry runs, otherwise the HF dataset."""
    from .config import HF_DATASET
    spec = os.environ.get("DPF_STORE", "")
    if spec.startswith("local:"):
        return LocalStore(spec[len("local:"):])
    return HFStore(HF_DATASET, token=os.environ.get("HF_TOKEN") or None)
