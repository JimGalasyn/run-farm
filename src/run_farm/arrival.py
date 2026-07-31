"""Artifact arrival: verify what landed, and publish it so nothing sees it partial.

Two halves of one rule -- **never let observable state be partial** -- applied to the
fetch side of a campaign.

**Verification.** A truncated 86 MB `field.npz` had a plausible size and a correct
`PK\\x03\\x04` magic number and failed only on open ("not a zip file"). It had been
copied by a fetcher polling every 120 s while the writer was still writing. It was
reported as banked on the strength of a directory listing. So: *do not claim a file is
good from a directory listing.* `verify_file` OPENS things.

For zip-family members that means CRC32, not a successful open. The distinction is
load-bearing, because the lazy readers callers actually use do not read member data
at all:

    corruption            zipfile.ZipFile(p)   testzip()
    truncated prefix      BadZipFile           BadZipFile
    flipped payload byte  SUCCEEDS             BAD: <member>

Coverage is honest about its own limits. Opaque formats (`.bin`, `.pt`, anything with
no verifier) carry no internal checksum, so nothing here can vouch for them; they are
counted `unverifiable` and reported as such. A report reading "all clear" over a
directory of opaque blobs would be the same overclaim in a different costume.

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
import gzip
import json
import os
import shutil
import tarfile
import zipfile
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path

# Suffixes that mean "a write was in progress here". Their presence is evidence of an
# interrupted save, not damage in itself -- an atomic writer leaves exactly this behind
# when it dies, with the real file untouched. Surfaced, never fatal.
RESIDUE_SUFFIXES = (".tmp", ".part", ".partial", ".crdownload", ".filepart")


@dataclasses.dataclass(frozen=True)
class ArrivalProblem:
    """One artifact that did not arrive intact.

    `fatal` separates "this data is unusable" from "this is a fault worth seeing".
    Residue and an empty fetch are the non-fatal cases: both are evidence about how a
    write went, not proof that what landed is unreadable.
    """

    path: str
    kind: str          # truncated | unreadable | empty | missing | tmp_residue
    detail: str
    fatal: bool = True

    def __str__(self) -> str:
        return f"[{self.kind}] {self.path}: {self.detail}"


# ---------------------------------------------------------------- verify ----
def _verify_zip(p: Path) -> ArrivalProblem | None:
    """CRC-verify every member. A .npz is a zip: a torn one keeps its header and
    loses its central directory, which is exactly why size and magic bytes both
    looked fine. `testzip()` streams the member data and checks the CRC32 the
    format already stores, which is what a lazy open does not do."""
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


def _verify_gzip(p: Path) -> ArrivalProblem | None:
    """Decompress fully; gzip's trailing CRC32 and length only verify on a complete
    read, so a truncated member is invisible until the last block."""
    try:
        with gzip.open(p, "rb") as f:
            while f.read(1 << 20):
                pass
    except (OSError, EOFError, zlib.error) as e:
        # OSError covers gzip.BadGzipFile; zlib.error is a raw inflate failure;
        # EOFError is the truncation case. Deliberately NOT bare Exception -- a check
        # that swallows everything cannot distinguish a corrupt file from a bug in
        # itself.
        return ArrivalProblem(str(p), "truncated",
                              f"gzip integrity failed ({type(e).__name__}: {e})")
    return None


def _verify_tar(p: Path) -> ArrivalProblem | None:
    """Walk every member and read it. tar has no per-member checksum over the DATA
    (only the header), so a full read is the strongest available check: it catches
    truncation, which is the failure mode that actually occurs."""
    try:
        with tarfile.open(p) as t:
            for member in t:
                if not member.isfile():
                    continue
                f = t.extractfile(member)
                if f is None:
                    continue
                while f.read(1 << 20):
                    pass
    except (tarfile.TarError, OSError, EOFError) as e:
        return ArrivalProblem(str(p), "truncated",
                              f"tar integrity failed ({type(e).__name__}: {e})")
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


#: suffix -> verifier. Anything absent here is counted `unverifiable` and reported as
#: such rather than quietly passing.
_VERIFIERS = {
    ".zip": _verify_zip, ".npz": _verify_zip, ".whl": _verify_zip,
    ".gz": _verify_gzip, ".gzip": _verify_gzip,
    ".tar": _verify_tar, ".tgz": _verify_tar,
    ".json": _verify_json,
    ".npy": _verify_npy,
}


def _verifier_for(p: Path):
    """The verifier for `p`, or None if this format carries nothing to check."""
    suffixes = [s.lower() for s in p.suffixes]
    # .tar.gz / .tar.bz2 are tar-checked, which subsumes the outer compression.
    if len(suffixes) >= 2 and suffixes[-2] == ".tar":
        return _verify_tar
    return _VERIFIERS.get(suffixes[-1] if suffixes else "")


def verify_file(path: str | Path) -> ArrivalProblem | None:
    """Open `path` and confirm it is intact. None means good.

    Falls back to a non-empty check for types with no structural verifier -- which is
    weak, and says so: for an unrecognised suffix a green result means only "bytes are
    present", not "the file is complete". `verify_report` counts those separately.
    """
    p = Path(path)
    if not p.exists():
        return ArrivalProblem(str(p), "missing", "does not exist")
    if p.is_dir():
        return None
    if p.stat().st_size == 0:
        return ArrivalProblem(str(p), "empty", "zero bytes")
    verifier = _verifier_for(p)
    return verifier(p) if verifier else None


@dataclasses.dataclass
class ArrivalReport:
    """What a tree verification found, with the unverifiable counted rather than
    folded into the pass."""

    root: Path
    problems: list[ArrivalProblem] = dataclasses.field(default_factory=list)
    verified: list[Path] = dataclasses.field(default_factory=list)
    unverifiable: list[Path] = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(p.fatal for p in self.problems)

    @property
    def fatal(self) -> list[ArrivalProblem]:
        return [p for p in self.problems if p.fatal]

    def render(self) -> str:
        head = (f"{self.root}: {len(self.verified)} verified, "
                f"{len(self.unverifiable)} unverifiable, "
                f"{len(self.fatal)} fatal, "
                f"{len(self.problems) - len(self.fatal)} warning(s)")
        return "\n".join([head] + [f"  {p}" for p in self.problems])

    def summary(self) -> str:
        """One line, for a LegResult detail field."""
        if self.ok:
            return f"{len(self.verified)} verified, {len(self.unverifiable)} opaque"
        first = self.fatal[0]
        more = f" (+{len(self.fatal) - 1} more)" if len(self.fatal) > 1 else ""
        return f"{Path(first.path).name}: [{first.kind}] {first.detail}{more}"


def _is_residue(p: Path) -> bool:
    name = p.name.lower()
    return any(name.endswith(s) for s in RESIDUE_SUFFIXES)


def verify_report(root: str | Path, *, require: Sequence[str] = (),
                  patterns: Iterable[str] = ("*",)) -> ArrivalReport:
    """Verify every file under `root`, plus a list of paths that MUST be present.

    Also flags partial-write residue (`*.tmp` and friends): a surviving temp file is
    evidence that a writer died mid-write, which is worth knowing even when everything
    else verifies. Residue is non-fatal -- an atomic writer leaves exactly this behind
    with the real file intact.

    An EMPTY tree is reported, but as a warning rather than a failure. A fetch that
    produced nothing is worth seeing -- it is the `done_when` trap in another costume,
    where absence reads as success. But `done_when` is the CALLER's declaration of what
    complete means, and a leg whose marker is the fetch directory itself is legitimately
    satisfied by an empty one. Overriding that from here would redefine a caller's
    contract in the name of integrity, which is not this module's business.
    """
    root = Path(root)
    report = ArrivalReport(root=root)
    if not root.exists():
        report.problems.append(ArrivalProblem(str(root), "missing",
                                              "output root does not exist"))
        return report
    if not root.is_dir():
        # rglob on a plain file yields nothing, which would otherwise read as an
        # empty (non-fatal) tree -- a check that cannot fail.
        report.problems.append(ArrivalProblem(str(root), "missing",
                                              "output root is not a directory"))
        return report

    for rel in require:
        if not (root / rel).exists():
            report.problems.append(ArrivalProblem(str(root / rel), "missing",
                                                  "required artifact absent"))
    for pat in patterns:
        for p in sorted(root.rglob(pat)):
            if not p.is_file():
                continue
            if _is_residue(p):
                report.problems.append(ArrivalProblem(
                    str(p), "tmp_residue",
                    "leftover temp file -- a writer died mid-write here; the "
                    "corresponding final file may be from an earlier attempt",
                    fatal=False))
                continue
            prob = verify_file(p)
            if prob is not None:
                report.problems.append(prob)
            (report.verified if _verifier_for(p) else report.unverifiable).append(p)

    if not report.verified and not report.unverifiable:
        report.problems.append(ArrivalProblem(
            str(root), "empty",
            "fetch produced no files -- nothing was transferred. If this leg must "
            "not accept an empty result, give it a precise done_when (e.g. "
            "'out/manifest.json') rather than the fetch directory itself.",
            fatal=False))
    return report


def verify_tree(root: str | Path, *, require: Sequence[str] = (),
                patterns: Iterable[str] = ("*",)) -> list[ArrivalProblem]:
    """The problems from `verify_report`, for callers that only want the list."""
    return verify_report(root, require=require, patterns=patterns).problems


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
