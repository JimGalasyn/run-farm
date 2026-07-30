"""Payload-closure checks, each a reconstruction of a failure that cost a rental.

These are written as NEGATIVE CONTROLS first: every test that asserts the validator
passes is paired with one that breaks the payload in the specific way a real campaign
broke it and asserts the validator FAILS. A preflight suite whose checks cannot fail
is decoration, and that is exactly how three of the original failures got past their
own verification.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from run_farm.payload import (PayloadError, PayloadSpec, Problem, require_flat,
                              validate_flat)


@pytest.fixture
def repo(tmp_path):
    """A repo laid out over THREE directories, like the one that broke twice.

        pkg/entry.py          the shipped entrypoint
        sibling.py            resolved by entry via _HERE.parent  <-- breaks when flat
        deep/lazy_dep.py      imported lazily inside a function   <-- missed entirely
    """
    (tmp_path / "pkg").mkdir()
    (tmp_path / "deep").mkdir()

    # entry.py resolves the sibling correctly -- probing BOTH layouts, which is how
    # the real fix was written -- so a complete payload passes.
    (tmp_path / "pkg" / "entry.py").write_text(textwrap.dedent('''
        from pathlib import Path
        _HERE = Path(__file__).resolve().parent

        def _find(name):
            for d in (_HERE.parent, _HERE):
                if (d / name).exists():
                    return d / name
            return _HERE.parent / name          # preserve the original error surface

        ENGINE_FILES = [_find("sibling.py")]

        def engine_sha():
            return sum(len(p.read_bytes()) for p in ENGINE_FILES)

        def scored():
            from lazy_dep import score      # lazy: never seen at import time
            return score()
    '''))

    # entry_flat_bug.py is the ORIGINAL defect: parent-relative only. At home the
    # sibling is in the parent dir; flattened into the worker cwd, _HERE.parent is
    # the temp dir's parent and the read raises at STARTUP, not at import.
    (tmp_path / "pkg" / "entry_flat_bug.py").write_text(textwrap.dedent('''
        from pathlib import Path
        _HERE = Path(__file__).resolve().parent
        ENGINE_FILES = [_HERE.parent / "sibling.py"]

        def engine_sha():
            return sum(len(p.read_bytes()) for p in ENGINE_FILES)
    '''))
    (tmp_path / "sibling.py").write_text("VALUE = 'sibling'\n")
    (tmp_path / "deep" / "lazy_dep.py").write_text("def score():\n    return 7\n")
    return tmp_path


def _spec(repo, files, **kw):
    return PayloadSpec(files=tuple(repo / f for f in files), **kw)


# --------------------------------------------------------------- the happy path
def test_complete_payload_passes(repo):
    spec = _spec(repo, ["pkg/entry.py", "sibling.py", "deep/lazy_dep.py"],
                 imports=("entry", "sibling", "lazy_dep"),
                 startup="import entry; print(entry.engine_sha())")
    assert validate_flat(spec) == []


def test_expect_matches_a_known_startup_value(repo):
    want = str(len((repo / "sibling.py").read_bytes()))
    spec = _spec(repo, ["pkg/entry.py", "sibling.py"], imports=("entry",),
                 startup="import entry; print(entry.engine_sha())", expect=want)
    assert validate_flat(spec) == []


# ------------------------------------------------- failure 1: one file short
def test_parent_relative_path_is_caught_by_startup_not_by_import(repo):
    """The exact original bug, and the reason `startup` exists.

    `import entry_flat_bug` SUCCEEDS -- the read happens inside engine_sha() -- so an
    IMPORT-ONLY check passes and the rental still dies at exit=1. Shipping the sibling
    does not help either: the code looks in the wrong directory once flattened."""
    ship = ["pkg/entry_flat_bug.py", "sibling.py"]

    spec = _spec(repo, ship, imports=("entry_flat_bug",))
    assert validate_flat(spec) == [], "import alone cannot see this bug"

    spec = _spec(repo, ship, imports=("entry_flat_bug",),
                 startup="import entry_flat_bug as e; print(e.engine_sha())")
    problems = validate_flat(spec)
    assert [p.kind for p in problems] == ["startup"]
    assert "FileNotFoundError" in problems[0].detail


# ------------------------------------------------ failure 2: lazy transitive dep
def test_lazily_imported_dep_is_caught_when_declared(repo):
    spec = _spec(repo, ["pkg/entry.py", "sibling.py"],
                 imports=("entry", "sibling", "lazy_dep"))
    problems = validate_flat(spec)
    assert [p.kind for p in problems] == ["import"]
    assert "lazy_dep" in problems[0].detail

    # and it is caught by exercising the code path, even if undeclared
    spec = _spec(repo, ["pkg/entry.py", "sibling.py"], imports=("entry",),
                 startup="import entry; print(entry.scored())")
    assert [p.kind for p in validate_flat(spec)] == ["startup"]


# --------------------------------------------- the check that proved nothing
def test_import_resolving_outside_the_flat_dir_is_a_problem(tmp_path, monkeypatch):
    """A module satisfied by the ambient environment rather than the payload. This
    is the mistake that made a fresh-clone test 'pass' while testing nothing: the
    import resolved through an editable install to the ORIGINAL tree."""
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "ambient.py").write_text("X = 1\n")
    (tmp_path / "shipped.py").write_text("Y = 2\n")
    monkeypatch.setenv("PYTHONPATH", str(outside))

    spec = PayloadSpec(files=(tmp_path / "shipped.py",),
                       imports=("shipped", "ambient"))
    problems = validate_flat(spec)
    assert [p.kind for p in problems] == ["import"]
    assert "OUTSIDE the flat dir" in problems[0].detail


# ------------------------------------------------------- flattening collisions
def test_same_basename_from_two_dirs_is_caught(tmp_path):
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "util.py").write_text(f"WHICH = {d!r}\n")
    spec = PayloadSpec(files=(tmp_path / "a" / "util.py", tmp_path / "b" / "util.py"))
    problems = validate_flat(spec)
    assert [p.kind for p in problems] == ["name_collision"]
    assert "keeps only one" in problems[0].detail


def test_missing_file_reported_before_staging(tmp_path):
    spec = PayloadSpec(files=(tmp_path / "nope.py",), imports=("nope",))
    problems = validate_flat(spec)
    assert [p.kind for p in problems] == ["missing_file"]   # not also an import error


# -------------------------------------------------------------- expect mismatch
def test_startup_that_runs_but_computes_the_wrong_thing(repo):
    spec = _spec(repo, ["pkg/entry.py", "sibling.py"], imports=("entry",),
                 startup="import entry; print(entry.engine_sha())", expect="999999")
    problems = validate_flat(spec)
    assert [p.kind for p in problems] == ["mismatch"]


# ----------------------------------------------------------------- launch gate
def test_require_flat_raises_with_every_problem_listed(repo):
    spec = _spec(repo, ["pkg/entry.py"], imports=("entry", "lazy_dep"),
                 startup="import entry; print(entry.engine_sha())")
    with pytest.raises(PayloadError) as ei:
        require_flat(spec)
    kinds = {p.kind for p in ei.value.problems}
    assert kinds == {"import", "startup"}, "should report ALL problems, not the first"
    assert "lazy_dep" in str(ei.value)


def test_problem_is_frozen_and_printable():
    import dataclasses
    p = Problem("import", "x: ImportError: y")
    assert str(p) == "[import] x: ImportError: y"
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.kind = "other"                                    # type: ignore[misc]


# ------------------------------------------------------- launch-command parity
def test_env_reaches_startup_and_its_absence_is_detectable(tmp_path):
    """The launch command exports ENGINE_COMMIT before running the entrypoint. A
    validator that cannot set env validates a DIFFERENT startup than the one that
    runs -- and the difference is not a crash, it is a silently wrong provenance
    hash. Verified against the real B2 payload: flat with no ENGINE_COMMIT the
    engine sha degrades to 'nogit:<content hash>', and with it exported the sha is
    bit-identical to the value computed at home.
    """
    (tmp_path / "prov.py").write_text(
        "import os\n"
        "def sha():\n"
        "    c = os.environ.get('ENGINE_COMMIT')\n"
        "    return c[:12] if c else 'nogit:fallback'\n")
    spec_args = dict(files=(tmp_path / "prov.py",),
                     startup="import prov; print(prov.sha())")

    # absent -> the fallback fires and `expect` catches it
    problems = validate_flat(PayloadSpec(**spec_args, expect="deadbeefcafe"))
    assert [p.kind for p in problems] == ["mismatch"]
    assert "nogit:fallback" in problems[0].detail

    # exported -> matches what home computes
    assert validate_flat(PayloadSpec(
        **spec_args, expect="deadbeefcafe",
        env=(("ENGINE_COMMIT", "deadbeefcafe0000"),))) == []


def test_env_does_not_wipe_the_ambient_environment(tmp_path):
    """Setting one variable must OVERLAY os.environ, not replace it: dropping PATH
    or HOME breaks interpreters in ways that read as a payload defect."""
    (tmp_path / "amb.py").write_text(
        "import os; print('PATH' in os.environ and bool(os.environ['PATH']))\n")
    assert validate_flat(PayloadSpec(
        files=(tmp_path / "amb.py",), startup="import amb",
        env=(("SOME_FLAG", "1"),), expect="True")) == []
