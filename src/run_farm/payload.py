"""Payload closure: does what you SHIP actually run where it LANDS? (free, local)

A `FleetLeg.ship` tuple is a list of files copied FLAT into the worker's working
directory. That flattening is the whole problem: a repo laid out over several
directories imports fine at home and fails on the box, and you pay a rental to find
out. Every check here runs locally in a temp directory for zero dollars.

Three real failures this module exists to catch, all from one 2026-07 campaign:

  1. A payload was one file short. The entrypoint resolved a sibling via
     `_HERE.parent / "gpe_vortex_topology.py"`; flattened into /workspace that is
     `/gpe_vortex_topology.py`, so a `read_bytes()` at startup raised
     FileNotFoundError and the batteries exited before running a single leg.
     Cost: one rental, and the log said only `exit=1`.
  2. It was short a SECOND file. `gpe_vortex_topology.knot_determinants()` imports
     `core_knot_id`, which lived in a third directory and was never shipped. This
     one surfaced on a local flat-layout run, for free, after the first had cost
     money -- which is the entire argument for this module.
  3. A smoke test SHIPPED the entrypoint and then ran `hostname` instead of it, so
     it validated the transport and not the payload, and sailed past a startup
     failure sitting in the file it had just copied.

So: importing is not enough, and shipping is not enough. `imports` must resolve
INSIDE the flat directory (a module that resolves to your source tree has proven
nothing -- that mistake was made too), and `startup` must actually execute the
entrypoint's real startup path.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class Problem:
    """One reason this payload would not run on a worker."""

    kind: str          # missing_file | name_collision | import | startup | mismatch
    detail: str

    def __str__(self) -> str:                      # pragma: no cover - display only
        return f"[{self.kind}] {self.detail}"


@dataclasses.dataclass(frozen=True)
class PayloadSpec:
    """What ships, and what must work once it is flat.

    files    paths copied flat into the worker cwd -- i.e. `FleetLeg.ship`.
    imports  module names that must import IN the flat dir and resolve to a file
             INSIDE it. Name every transitive local module, including ones imported
             lazily inside a function: failure 2 above was a lazy import.
    startup  python source run in the flat dir. Use the entrypoint's real startup,
             not a stand-in; failure 1 raised at startup, not at import.
    expect   if given, `startup`'s last stdout line must equal this. Catches a
             payload that runs but computes something different from home (e.g. a
             provenance hash resolved from the wrong place).
    env      (name, value) pairs exported for both probe and startup. If the launch
             command exports something, set it here -- otherwise you are validating
             a DIFFERENT startup than the one that will run. The B2 campaign exports
             ENGINE_COMMIT precisely because the payload ships flat onto a box with
             no git repo, and without it provenance silently degrades to a content
             hash: a payload that runs and certifies the wrong tree.
    """

    files: tuple[str | Path, ...]
    imports: tuple[str, ...] = ()
    startup: str | None = None
    expect: str | None = None
    env: tuple[tuple[str, str], ...] = ()


_PROBE = r"""
import json, sys, pathlib
here = pathlib.Path.cwd().resolve()
out = {"imported": {}, "errors": []}
for name in json.loads(sys.argv[1]):
    try:
        mod = __import__(name)
        f = getattr(mod, "__file__", None)
        out["imported"][name] = str(pathlib.Path(f).resolve()) if f else "<builtin>"
    except BaseException as e:
        out["errors"].append([name, type(e).__name__, str(e)[:400]])
print("__PROBE__" + json.dumps(out))
"""


def validate_flat(spec: PayloadSpec, *, python: str | None = None,
                  keep: bool = False) -> list[Problem]:
    """Copy `spec.files` flat into a temp dir and check they run there.

    Returns [] when the payload is sound. Never raises for payload defects -- they
    are the result, not an exception -- so a caller can print them all at once
    instead of discovering them one rental at a time.

    Runs in a SUBPROCESS with cwd set to the flat dir, because an in-process import
    would be satisfied by the caller's own sys.path and pass while proving nothing.
    """
    python = python or sys.executable
    env = {**os.environ, **dict(spec.env)} if spec.env else None
    problems: list[Problem] = []
    resolved: list[Path] = []
    seen: dict[str, Path] = {}

    for f in spec.files:
        p = Path(f).expanduser()
        if not p.exists():
            problems.append(Problem("missing_file", f"{p} does not exist"))
            continue
        # Flattening collapses directories, so two files with the same basename in
        # different dirs silently become one on the worker.
        if p.name in seen and seen[p.name].resolve() != p.resolve():
            problems.append(Problem(
                "name_collision",
                f"{p.name} ships from both {seen[p.name].parent} and {p.parent}; "
                "flattening keeps only one"))
        seen[p.name] = p
        resolved.append(p)

    if problems:
        return problems                            # nothing to stage; report early

    tmp = Path(tempfile.mkdtemp(prefix="run_farm_payload_"))
    try:
        for p in resolved:
            shutil.copy2(p, tmp / p.name)

        if spec.imports:
            r = subprocess.run([python, "-c", _PROBE, json.dumps(list(spec.imports))],
                               cwd=tmp, capture_output=True, text=True, timeout=300,
                               env=env)
            line = next((l for l in r.stdout.splitlines()
                         if l.startswith("__PROBE__")), None)
            if line is None:
                problems.append(Problem(
                    "import", f"probe did not report (rc={r.returncode}): "
                              f"{(r.stderr or r.stdout).strip()[:400]}"))
            else:
                got = json.loads(line[len("__PROBE__"):])
                for name, kind, msg in got["errors"]:
                    problems.append(Problem("import", f"{name}: {kind}: {msg}"))
                for name, path in got["imported"].items():
                    if path != "<builtin>" and not path.startswith(str(tmp)):
                        problems.append(Problem(
                            "import",
                            f"{name} imported from {path}, OUTSIDE the flat dir -- "
                            "it resolved via the ambient environment, so this check "
                            "proved nothing about the payload"))

        if spec.startup:
            r = subprocess.run([python, "-c", spec.startup], cwd=tmp,
                               capture_output=True, text=True, timeout=900, env=env)
            if r.returncode != 0:
                tail = (r.stderr or r.stdout).strip().splitlines()[-6:]
                problems.append(Problem(
                    "startup", f"rc={r.returncode}: " + " | ".join(tail)))
            elif spec.expect is not None:
                lines = [l for l in r.stdout.strip().splitlines() if l.strip()]
                actual = lines[-1].strip() if lines else ""
                if actual != spec.expect.strip():
                    problems.append(Problem(
                        "mismatch",
                        f"startup printed {actual!r}, expected {spec.expect.strip()!r}"))
    finally:
        if keep:
            print(f"payload staged at {tmp}", file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    return problems


def require_flat(spec: PayloadSpec, **kw) -> None:
    """validate_flat, but raise on any problem. For use as a launch gate."""
    problems = validate_flat(spec, **kw)
    if problems:
        raise PayloadError(problems)


class PayloadError(RuntimeError):
    def __init__(self, problems: list[Problem]):
        self.problems = problems
        super().__init__("payload would not run on a worker:\n  "
                         + "\n  ".join(str(p) for p in problems))
