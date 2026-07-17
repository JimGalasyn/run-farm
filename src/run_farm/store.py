"""A shared object-store backend for the campaign registry + event sink.

`ObjectStoreRunRegistry` (A/B) and `ObjectStoreEventSink` (C) implement the same
protocols as the local `FileRunRegistry`/`JsonlEventSink`, but over a key-value
**`BlobStore`** instead of a local directory. Point every executor at one store
(an S3/GCS/R2 bucket) and the campaign gains what the per-executor local dirs
can't give: a single source of truth across clouds -- global `is_complete`
(dedup + cross-provider/-restart resume) and one place to read results and the
streamed event ledger from. No new seam; a second backend of the existing A/B/C
protocols (CAMPAIGN.md's "RunRegistry on a shared object store").

Design note: every record is its **own blob** (`events/<time_ns>-<uuid>.json`,
`_manifest/<run>.json`), never an appended-to file. Object stores have no atomic
append, and one-blob-per-record means concurrent writers -- the whole point of a
shared store -- never race on a single key (the key carries a UUID so even
same-tick writers don't collide). Reading a stream is list + sort.

`boto3` is an optional dependency, imported only by `S3BlobStore`.
"""

from __future__ import annotations

import io
import json
import time
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import numpy as np

from run_farm.protocols import RunHandle, State
from run_farm.protocols import RunConfig


# ----------------------------------------------------------------- BlobStore --
@runtime_checkable
class BlobStore(Protocol):
    """A minimal key -> bytes store. Keys are '/'-delimited, store-relative.

    The whole object-store dependency surface: four methods, so a new backend
    (GCS, R2, MinIO, Azure) is a tiny adapter. Implementations must make `put`
    durable before it returns and `list(prefix)` return every key with that
    prefix (store-relative, paginated internally).
    """

    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, data: bytes) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str) -> list[str]: ...


class MemoryBlobStore:
    """In-process `BlobStore` (a dict). For tests and single-process campaigns."""

    def __init__(self) -> None:
        self._d: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._d.get(key)

    def put(self, key: str, data: bytes) -> None:
        self._d[key] = bytes(data)

    def exists(self, key: str) -> bool:
        return key in self._d

    def list(self, prefix: str) -> list[str]:
        return [k for k in self._d if k.startswith(prefix)]


class S3BlobStore:  # pragma: no cover  (live S3/boto3; not exercised in CI)
    """`BlobStore` over any S3-compatible service (AWS S3, R2, MinIO, GCS-S3).

    `prefix` scopes all keys under a bucket sub-path and is stripped from
    `list`, so the registry works in one store-relative key space regardless of
    where the bucket roots it. Credentials come from the standard boto3 chain
    (env / ~/.aws / instance role); pass `client` to inject a configured one
    (e.g. a non-AWS endpoint_url). `boto3` is imported lazily here only.
    """

    def __init__(self, bucket: str, prefix: str = "", *, client=None,
                 **session_kwargs) -> None:
        import boto3
        self._s3 = client or boto3.client("s3", **session_kwargs)
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _full(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str) -> bytes | None:
        from botocore.exceptions import ClientError
        try:
            obj = self._s3.get_object(Bucket=self.bucket, Key=self._full(key))
            return obj["Body"].read()
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    def put(self, key: str, data: bytes) -> None:
        self._s3.put_object(Bucket=self.bucket, Key=self._full(key), Body=data)

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._full(key))
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> list[str]:
        full = self._full(prefix)
        cut = len(self.prefix) + 1 if self.prefix else 0
        out: list[str] = []
        token = None
        while True:
            kw = {"Bucket": self.bucket, "Prefix": full}
            if token:
                kw["ContinuationToken"] = token
            resp = self._s3.list_objects_v2(**kw)
            out.extend(o["Key"][cut:] for o in resp.get("Contents", []))
            if not resp.get("IsTruncated"):
                return out
            token = resp.get("NextContinuationToken")


# --------------------------------------------------- checkpoint (de)serialize --
# Same .npz layout as runs.save_checkpoint, but to/from bytes for a blob store.
def _dump_checkpoint(state: State, config: RunConfig, step: int) -> bytes:
    buf = io.BytesIO()
    arrays = {f"state__{k}": np.asarray(v) for k, v in state.items()}
    np.savez_compressed(buf, __config__=config.to_json(), __step__=step, **arrays)
    return buf.getvalue()


def _load_checkpoint(data: bytes) -> tuple[State, int]:
    with np.load(io.BytesIO(data), allow_pickle=False) as d:
        step = int(d["__step__"])
        state = {k[len("state__"):]: jnp.asarray(d[k])
                 for k in d.files if k.startswith("state__")}
    return state, step


def _read_json_blobs(store: "BlobStore", keys: list[str]) -> list[dict]:
    """Read + JSON-parse each key, skipping any whose blob is gone.

    `list()` then `get()` is not atomic: on an eventually-consistent backend, or
    if a key is pruned between the two calls, `get()` can return None. Skip the
    vanished row rather than crash the whole ledger/manifest read on a TypeError.
    """
    out = []
    for k in keys:
        blob = store.get(k)
        if blob is not None:
            out.append(json.loads(blob))
    return out


# ----------------------------------------------------------------- A + B (P4) --
class ObjectStoreRunRegistry:
    """`RunRegistry` over a `BlobStore` -- config-hashed runs + full-state
    checkpoints, shared across every writer pointed at the same store."""

    def __init__(self, store: BlobStore, base: str = "runs"):
        self.store = store
        self.base = base.rstrip("/")

    def _key(self, name: str, *parts: str) -> str:
        return "/".join([self.base, name, *parts])

    def register(self, config: RunConfig) -> RunHandle:
        name = config.run_name()
        cfg_key = self._key(name, "config.json")
        if not self.store.exists(cfg_key):           # idempotent (A)
            self.store.put(cfg_key, (config.to_json() + "\n").encode())
            self.store.put(
                f"{self.base}/_manifest/{name}.json",
                json.dumps({"run": name,
                            "config": json.loads(config.to_json())}).encode())
        return RunHandle(config=config, dir=Path(f"{self.base}/{name}"), name=name)

    def is_complete(self, handle: RunHandle) -> bool:
        return self.store.exists(self._key(handle.name, "DONE.json"))

    def load(self, handle: RunHandle) -> tuple[State, int] | None:
        data = self.store.get(self._key(handle.name, "checkpoint.npz"))
        return None if data is None else _load_checkpoint(data)

    def save(self, handle: RunHandle, state: State, step: int) -> None:
        self.store.put(self._key(handle.name, "checkpoint.npz"),
                       _dump_checkpoint(state, handle.config, step))

    def finish(self, handle: RunHandle, result: dict) -> None:
        self.store.put(self._key(handle.name, "DONE.json"),
                       (json.dumps(result, sort_keys=True) + "\n").encode())

    def manifest(self) -> list[dict]:
        """Every registered run (one manifest blob each)."""
        return _read_json_blobs(
            self.store, sorted(self.store.list(f"{self.base}/_manifest/")))


# -------------------------------------------------------------- C (P6, P7) --
class ObjectStoreEventSink:
    """`EventSink` over a `BlobStore`: one blob per emitted record / triggered
    capture, so concurrent writers never contend. `events(name)` reads the
    streamed ledger back -- the cross-provider live view."""

    def __init__(self, store: BlobStore, base: str = "runs"):
        self.store = store
        self.base = base.rstrip("/")

    @staticmethod
    def _blob_key() -> str:
        """A sortable, collision-resistant per-record key.

        A shared sequence counter (or `len(list(...))`) races: two writers for
        the same run can pick the same number and silently overwrite each other.
        A nanosecond timestamp keeps blobs in emission order under a lexical sort
        (zero-padded to fixed width), and a UUID suffix makes a same-tick
        collision between writers effectively impossible."""
        return f"{time.time_ns():020d}-{uuid.uuid4().hex}"

    def emit(self, handle: RunHandle, record: dict) -> None:
        self.store.put(
            f"{self.base}/{handle.name}/events/{self._blob_key()}.json",
            json.dumps(record, sort_keys=True).encode())

    def trigger(self, handle: RunHandle, state: State, reason: str) -> None:
        buf = io.BytesIO()
        np.savez_compressed(buf, __reason__=reason,
                            **{k: np.asarray(v) for k, v in state.items()})
        self.store.put(
            f"{self.base}/{handle.name}/triggered/{self._blob_key()}.npz",
            buf.getvalue())

    def close(self, handle: RunHandle) -> None:
        pass                                          # each emit is already durable

    def events(self, name: str) -> list[dict]:
        """The streamed ledger for a run, in order -- read it any time, from any
        process pointed at the store (the live cross-provider view)."""
        keys = sorted(self.store.list(f"{self.base}/{name}/events/"))
        return _read_json_blobs(self.store, keys)
