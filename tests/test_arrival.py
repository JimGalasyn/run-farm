"""Arrival integrity + the marker-last publication contract.

Every check here is paired with the corruption it is supposed to catch, because the
original failure was a file that passed every cheap test -- plausible size, correct
magic bytes -- and failed only on open.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

from run_farm.arrival import (ArrivalProblem, atomic_write, publish, verify_file,
                              verify_tree)


def _npz(path: Path, n: int = 200) -> None:
    np.savez(str(path), a=np.arange(n, dtype=np.float64), b=np.ones((n, 3)))


# ------------------------------------------------------- the historical tear
def test_truncated_npz_is_caught_but_looks_fine_by_every_cheap_test(tmp_path):
    """The exact 86 MB `field.npz` failure. A mid-write copy keeps the PK header and
    loses the central directory: size is plausible, magic bytes are right, and it
    fails only on open. This is why a directory listing is not evidence."""
    good = tmp_path / "field.npz"
    _npz(good, 500)
    torn = tmp_path / "torn.npz"
    data = good.read_bytes()
    torn.write_bytes(data[: int(len(data) * 0.8)])          # lose the EOCD

    # the cheap tests a listing would do -- both PASS on the torn file
    assert torn.stat().st_size > 0
    assert torn.read_bytes()[:4] == b"PK\x03\x04", "magic bytes still correct"

    assert verify_file(good) is None                         # negative control
    prob = verify_file(torn)
    assert prob is not None and prob.kind == "truncated"
    assert "zip" in prob.detail.lower()


def test_truncated_json_manifest_is_caught(tmp_path):
    good, torn = tmp_path / "m.json", tmp_path / "t.json"
    good.write_text(json.dumps({"Lk": -3.0, "nseg": 12}))
    torn.write_text(json.dumps({"Lk": -3.0, "nseg": 12})[:-4])
    assert verify_file(good) is None
    assert verify_file(torn).kind == "truncated"


def test_empty_and_missing_are_distinguished(tmp_path):
    (tmp_path / "zero.npz").write_bytes(b"")
    assert verify_file(tmp_path / "zero.npz").kind == "empty"
    assert verify_file(tmp_path / "nope.npz").kind == "missing"


def test_unrecognised_suffix_only_claims_bytes_are_present(tmp_path):
    """A weak check that says so: no structural verifier for .log, so a non-empty
    file passes. Documented so a green result is not over-read."""
    p = tmp_path / "progress.log"
    p.write_text("partial line without a newline")
    assert verify_file(p) is None


def test_verify_tree_flags_required_absent_and_tmp_residue(tmp_path):
    _npz(tmp_path / "field.npz")
    (tmp_path / "field.npz.tmp").write_bytes(b"half a file")   # writer died here
    problems = verify_tree(tmp_path, require=("manifest.json",))
    kinds = {p.kind for p in problems}
    assert kinds == {"missing", "tmp_residue"}
    assert any("manifest.json" in p.path for p in problems)


# ------------------------------------------------- the publication contract
def test_marker_is_published_last(tmp_path):
    """THE contract. A completion marker must be the last byte written, or it is not
    a completion marker."""
    staging, dest = tmp_path / "stage", tmp_path / "dest"
    staging.mkdir()
    _npz(staging / "field.npz")
    (staging / "manifest.json").write_text("{}")

    order = []
    real_replace = os.replace

    def spy(src, dst):
        order.append(Path(dst).name)
        return real_replace(src, dst)

    import run_farm.arrival as arrival
    orig, arrival.os.replace = arrival.os.replace, spy
    try:
        publish(staging, dest, marker="manifest.json")
    finally:
        arrival.os.replace = orig

    assert order[-1] == "manifest.json", f"marker not last: {order}"
    assert (dest / "field.npz").exists() and (dest / "manifest.json").exists()


def test_marker_inside_a_fetched_dir_is_still_published_last(tmp_path):
    """done_when="out_kick/manifest.json": the marker is nested, so the whole
    directory it lives in must go last, and within it the marker last again."""
    staging, dest = tmp_path / "stage", tmp_path / "dest"
    (staging / "out_kick").mkdir(parents=True)
    _npz(staging / "out_kick" / "field.npz")
    (staging / "out_kick" / "manifest.json").write_text("{}")
    (staging / "other.txt").write_text("x")

    order = []
    import run_farm.arrival as arrival
    real = arrival.os.replace
    arrival.os.replace = lambda s, d: (order.append(Path(d).name), real(s, d))[1]
    try:
        publish(staging, dest, marker="out_kick/manifest.json")
    finally:
        arrival.os.replace = real
    assert order[-1] == "manifest.json", order
    assert order.index("other.txt") < order.index("manifest.json")


def test_publish_reports_damage_without_dropping_the_file(tmp_path):
    """A corrupt artifact you can inspect beats one silently discarded."""
    staging, dest = tmp_path / "stage", tmp_path / "dest"
    staging.mkdir()
    good = tmp_path / "src.npz"
    _npz(good)
    (staging / "field.npz").write_bytes(good.read_bytes()[:100])   # torn
    problems = publish(staging, dest, marker="")
    assert [p.kind for p in problems] == ["truncated"]
    assert (dest / "field.npz").exists(), "damaged file must still be published"


def test_publish_removes_staging_and_overwrites_an_existing_dest(tmp_path):
    staging, dest = tmp_path / "stage", tmp_path / "dest"
    staging.mkdir(); dest.mkdir()
    (dest / "field.npz").write_bytes(b"old")
    _npz(staging / "field.npz")
    publish(staging, dest)
    assert not staging.exists()
    assert verify_file(dest / "field.npz") is None       # the NEW one, and intact


# --------------------------------------------------------------- atomic_write
def test_atomic_write_is_never_observable_half_written(tmp_path):
    """A reader either sees the previous file or the complete new one."""
    path = tmp_path / "out.json"
    path.write_text('{"gen": 1}')
    seen = []

    def writer(f):
        seen.append(json.loads(path.read_text()))        # mid-write, read the target
        f.write(b'{"gen": 2}')

    atomic_write(path, writer)
    assert seen == [{"gen": 1}], "target must still be the OLD file during the write"
    assert json.loads(path.read_text()) == {"gen": 2}
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_leaves_no_tmp_and_no_target_change_on_failure(tmp_path):
    path = tmp_path / "out.json"
    path.write_text("original")

    def boom(f):
        f.write(b"partial")
        raise RuntimeError("writer died")

    with pytest.raises(RuntimeError):
        atomic_write(path, boom)
    assert path.read_text() == "original", "previous file must survive"
    assert not list(tmp_path.glob("*.tmp")), "no residue"
