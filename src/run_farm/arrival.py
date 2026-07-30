"""Artifact arrival: verify what landed, and publish it so nothing sees it partial.

Two halves of one rule -- **never let observable state be partial** -- applied to the
fetch side of a campaign.

**Verification.** A truncated 86 MB `field.npz` had a plausible size and a correct
`PK\\x03\\x04` magic number and failed only on open ("not a zip file"). It had been
copied by a fetcher polling every 120 s while the writer was still writing. It was
reported as banked on the strength of a directory listing. So: *do not claim a file is
good from a directory listing.* `verify_file` OPENS things.

**Publication.** The same bug, in the other medium, in run-farm's own fetch path.
`scp -r` copies into the live leg directory in whatever order it likes, and a leg's
`done_when` marker is one of those files. If the marker lands before the payload and
the process dies mid-fetch, the leg dir now holds a marker with no results -- and the
next relaunch PRE-SKIPS it as complete, silently, forever. `publish` fixes that by
staging the fetch aside and moving the marker in **last**, so the marker's presence
implies the payload's presence.

That ordering is the entire contract: a completion marker must be the last byte
written, or it is not a completion marker.

Distinct from `farm.verify_shipment`, which checks a RunFn's returned *record* against
product-hash sidecars and engine-SHA attestation. That answers "is this result the one
the worker says it computed?"; this module answers "did the bytes survive the trip?".
A shipment can verify perfectly while the artifact it points at is torn.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import zipfile
from collections.abc import Iterable, Sequence
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class ArrivalProblem:
    """One artifact that did not arrive intact."""

    path: str
    kind: str          # truncated | unreadable | empty | missing | tmp_residue
    detail: str

    def __str__(self) -> str:
        return f"[{self.kind}] {self.path}: {self.detail}"


# ---------------------------------------------------------------- verify ----
def _verify_npz(p: Path) -> ArrivalProblem | None:
    """A .npz is a zip: a torn one keeps its header and loses its central
    directory, which is exactly why size and magic bytes both looked fine."""
    try:
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
            if bad is not None:
                return ArrivalProblem(str(p), "truncated",
                                      f"CRC failure in member {bad}")
            if not z.namelist():
                return ArrivalProblem(str(p), "empty", "zip contains no members")
    except zipfile.BadZipFile as e:
        return ArrivalProblem(str(p), "truncated",
                              f"not a valid zip ({e}) -- classic mid-write copy; "
                              "size and magic bytes can both still look correct")
    except OSError as e:
        return ArrivalProblem(str(p), "unreadable", str(e))
    return None


def _verify_json(p: Path) -> ArrivalProblem | None:
    try:
        text = p.read_text()
    except OSError as e:
        return ArrivalProblem(str(p), "unreadable", str(e))
    if not text.strip():
        return ArrivalProblem(str(p), "empty", "zero-length JSON")
    try:
        json.loads(text)
    except ValueError as e:
        return ArrivalProblem(str(p), "truncated",
                              f"invalid JSON ({e}) -- a half-written manifest")
    return None


def _verify_npy(p: Path) -> ArrivalProblem | None:
    try:
        import numpy as np
        np.load(str(p), allow_pickle=False, mmap_mode="r")
    except ImportError:                                       # pragma: no cover
        return None
    except Exception as e:                                    # noqa: BLE001
        return ArrivalProblem(str(p), "truncated", f"np.load failed: {e}")
    return None


_VERIFIERS = {".npz": _verify_npz, ".json": _verify_json, ".npy": _verify_npy}


def verify_file(path: str | Path) -> ArrivalProblem | None:
    """Open `path` and confirm it is intact. None means good.

    Falls back to a non-empty check for types with no structural verifier -- which is
    weak, and says so: for an unrecognised suffix a green result means only "bytes are
    present", not "the file is complete".
    """
    p = Path(path)
    if not p.exists():
        return ArrivalProblem(str(p), "missing", "does not exist")
    if p.is_dir():
        return None
    if p.stat().st_size == 0:
        return ArrivalProblem(str(p), "empty", "zero bytes")
    verifier = _VERIFIERS.get(p.suffix.lower())
    return verifier(p) if verifier else None


def verify_tree(root: str | Path, *, require: Sequence[str] = (),
                patterns: Iterable[str] = ("*",)) -> list[ArrivalProblem]:
    """Verify every file under `root`, plus a list of paths that MUST be present.

    Also flags leftover `*.tmp` files: a surviving temp file is evidence that a writer
    died mid-write, which is worth knowing even when everything else verifies.
    """
    root = Path(root)
    problems: list[ArrivalProblem] = []
    if not root.exists():
        return [ArrivalProblem(str(root), "missing", "output root does not exist")]

    for rel in require:
        if not (root / rel).exists():
            problems.append(ArrivalProblem(str(root / rel), "missing",
                                           "required artifact absent"))
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if not p.is_file():
                continue
            if p.name.endswith(".tmp"):
                problems.append(ArrivalProblem(
                    str(p), "tmp_residue",
                    "leftover temp file -- a writer died mid-write here; the "
                    "corresponding final file may be from an earlier attempt"))
                continue
            prob = verify_file(p)
            if prob is not None:
                problems.append(prob)
    return problems


# --------------------------------------------------------------- publish ----
def publish(staging: str | Path, dest: str | Path, *, marker: str = "",
            verify: bool = True) -> list[ArrivalProblem]:
    """Move everything from `staging` into `dest`, with `marker` moved LAST.

    The ordering is the contract. `dest` may be observed at any instant by a human, a
    monitor, or this campaign's own resume logic; at no instant may it contain the
    completion marker without the payload the marker claims is there.

    Per-entry `os.replace` is atomic, so each file appears whole or not at all. The
    directory as a whole is not atomic -- POSIX gives no multi-file atomic publish
    without a swap -- so partial *sets* remain possible after a crash. What is
    excluded is the failure that matters: a marker with no payload behind it.

    Returns any `ArrivalProblem`s found (when `verify`); publishing proceeds anyway,
    because a corrupt artifact you can inspect beats one silently dropped.
    """
    staging, dest = Path(staging), Path(dest)
    problems = verify_tree(staging) if verify else []
    dest.mkdir(parents=True, exist_ok=True)

    entries = sorted(staging.iterdir()) if staging.exists() else []
    marker_first = Path(marker).parts[0] if marker else ""
    # Anything the marker lives inside goes last, so the marker cannot precede the
    # payload even when it is nested (e.g. done_when="out_kick/manifest.json").
    ordered = ([e for e in entries if e.name != marker_first]
               + [e for e in entries if e.name == marker_first])

    for src in ordered:
        target = dest / src.name
        if src.is_dir():
            # os.replace refuses a non-empty destination dir, so merge file-by-file
            # (each file still lands atomically) and place the marker last within it.
            _merge_dir(src, target, marker)
        else:
            os.replace(src, target)
    shutil.rmtree(staging, ignore_errors=True)
    return problems


def _merge_dir(src: Path, dest: Path, marker: str) -> None:
    """Recursively merge `src` into `dest`, atomically per file, marker last."""
    dest.mkdir(parents=True, exist_ok=True)
    marker_name = Path(marker).name if marker else ""
    kids = sorted(src.iterdir())
    ordered = ([k for k in kids if k.name != marker_name]
               + [k for k in kids if k.name == marker_name])
    for k in ordered:
        if k.is_dir():
            _merge_dir(k, dest / k.name, marker)
        else:
            os.replace(k, dest / k.name)


def atomic_write(path: str | Path, write_fn, *, fsync: bool = True) -> None:
    """Write `path` via `<path>.tmp` + fsync + `os.replace`, never half-visible.

    `write_fn(handle)` receives a binary file object. fsync before the rename so the
    bytes are on disk, not merely in the page cache, before the name points at them --
    otherwise a power loss can leave a correctly-named, empty file.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(tmp, "wb") as f:
            write_fn(f)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
