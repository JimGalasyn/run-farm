"""Every check in `arrival` must be able to FAIL.

A validator that cannot fail is worse than no validator: it converts "unverified"
into "verified" and buys false confidence with real money. So each test here pairs a
good artifact with a deliberately damaged one and asserts BOTH outcomes.

`test_a_lazy_open_would_miss_this` is the load-bearing one. It pins the exact reason
`_verify_zip` calls `testzip()` rather than just opening the file -- if a future
stdlib makes a bare open strict, that test fails and the module's rationale needs
rewriting rather than silently becoming redundant.

`test_arrival.py` covers the publish/ordering half of the module; this file covers
verification depth, coverage honesty, and reporting.
"""
from __future__ import annotations

import gzip
import json
import tarfile
import zipfile

import pytest

from run_farm.arrival import ArrivalReport, _verifier_for, verify_file, verify_report


# --------------------------------------------------------------------------
# builders + the two ways a transfer goes wrong
# --------------------------------------------------------------------------
def make_zip(path, members=(("a.npy", b"x" * 4096), ("b.npy", b"y" * 4096))):
    with zipfile.ZipFile(path, "w") as z:
        for name, data in members:
            z.writestr(name, data)
    return path


def truncate(path, frac=0.5):
    """The mid-write copy: a valid prefix of a valid file."""
    raw = path.read_bytes()
    path.write_bytes(raw[:int(len(raw) * frac)])


def flip_payload_byte(path, offset=200):
    """Corrupt member DATA, leave the central directory intact.

    The case a lazy reader cannot see. This is why CRC verification exists.
    """
    raw = bytearray(path.read_bytes())
    start = raw.find(b"PK\x03\x04")
    raw[start + offset] ^= 0xFF
    path.write_bytes(bytes(raw))


@pytest.fixture
def leg_dir(tmp_path):
    d = tmp_path / "out_leg"
    d.mkdir()
    make_zip(d / "field.npz")
    (d / "manifest.json").write_text(json.dumps({"params": {"N": 8}, "traj": [1]}))
    return d


# --------------------------------------------------------------------------
# the control
# --------------------------------------------------------------------------
class TestIntactArtifactsPass:
    def test_clean_leg_is_ok(self, leg_dir):
        report = verify_report(leg_dir)
        assert report.ok, report.render()
        assert len(report.verified) == 2
        assert report.unverifiable == []

    def test_summary_is_one_line_when_ok(self, leg_dir):
        assert "\n" not in verify_report(leg_dir).summary()


# --------------------------------------------------------------------------
# zip / npz -- the failure the incident actually produced, and the one it didn't
# --------------------------------------------------------------------------
class TestZipIntegrity:
    def test_truncated_zip_is_caught(self, leg_dir):
        truncate(leg_dir / "field.npz")
        report = verify_report(leg_dir)
        assert not report.ok
        assert any(p.kind == "truncated" for p in report.problems)

    def test_payload_corruption_is_caught(self, leg_dir):
        flip_payload_byte(leg_dir / "field.npz")
        report = verify_report(leg_dir)
        assert not report.ok
        assert any("CRC failure" in p.detail for p in report.problems)

    def test_a_lazy_open_would_miss_this(self, tmp_path):
        """THE reason `_verify_zip` calls testzip(), as an executable claim.

        Opening a zip reads only the central directory. On a file whose payload
        bytes are corrupt but whose directory is intact, the open SUCCEEDS. A
        validator built on `try: ZipFile(p) except:` would pass this file.
        """
        p = make_zip(tmp_path / "f.npz")
        flip_payload_byte(p)

        with zipfile.ZipFile(p) as z:           # the naive check: passes
            assert z.namelist()                 # and even lists members

        with zipfile.ZipFile(p) as z:           # only a real read notices
            assert z.testzip() == "a.npy"

        problem = verify_file(p)
        assert _verifier_for(p) is not None
        assert problem is not None and "CRC failure" in problem.detail

    def test_bad_member_is_named(self, tmp_path):
        p = make_zip(tmp_path / "f.npz")
        flip_payload_byte(p)
        assert "a.npy" in verify_report(tmp_path).render()

    def test_empty_container_is_caught(self, tmp_path):
        with zipfile.ZipFile(tmp_path / "empty.npz", "w"):
            pass
        report = verify_report(tmp_path)
        assert not report.ok
        assert any("no members" in p.detail for p in report.problems)

    def test_non_zip_content_is_caught(self, leg_dir):
        """An HTML error page saved under a .npz name -- a real scp/proxy failure."""
        (leg_dir / "field.npz").write_bytes(b"<html>504 Gateway Timeout</html>")
        report = verify_report(leg_dir)
        assert not report.ok
        assert any("not a valid zip" in p.detail for p in report.problems)

    def test_empty_file_is_caught(self, leg_dir):
        (leg_dir / "field.npz").write_bytes(b"")
        assert not verify_report(leg_dir).ok

    def test_zip_and_whl_share_the_verifier(self, tmp_path):
        """.npz is not special-cased: any zip-family member gets CRC treatment."""
        for name in ("data.zip", "pkg.whl"):
            truncate(make_zip(tmp_path / name))
        report = verify_report(tmp_path)
        assert not report.ok and len(report.fatal) == 2


# --------------------------------------------------------------------------
# other verifiable formats
# --------------------------------------------------------------------------
class TestJson:
    def test_truncated_json_is_caught(self, leg_dir):
        truncate(leg_dir / "manifest.json", 0.4)
        report = verify_report(leg_dir)
        assert not report.ok
        assert any("invalid JSON" in p.detail for p in report.problems)

    def test_schema_is_not_this_module_s_business(self, leg_dir):
        """Transport integrity only. `{}` is intact and semantically useless, and
        saying otherwise would need a schema this layer must not have."""
        (leg_dir / "manifest.json").write_text("{}")
        assert verify_report(leg_dir).ok


class TestGzip:
    def test_intact_gzip_passes(self, tmp_path):
        with gzip.open(tmp_path / "log.gz", "wb") as f:
            f.write(b"z" * 100_000)
        assert verify_report(tmp_path).ok

    def test_truncated_gzip_is_caught(self, tmp_path):
        p = tmp_path / "log.gz"
        with gzip.open(p, "wb") as f:
            f.write(b"z" * 100_000)
        truncate(p, 0.5)
        report = verify_report(tmp_path)
        assert not report.ok
        assert any("gzip integrity failed" in x.detail for x in report.problems)


class TestTar:
    def _make(self, path, payload=b"q" * 100_000):
        src = path.parent / "member.bin"
        src.write_bytes(payload)
        with tarfile.open(path, "w:gz") as t:
            t.add(src, arcname="member.bin")
        src.unlink()
        return path

    def test_intact_tar_gz_passes(self, tmp_path):
        self._make(tmp_path / "bundle.tar.gz")
        assert verify_report(tmp_path).ok

    def test_truncated_tar_gz_is_caught(self, tmp_path):
        p = self._make(tmp_path / "bundle.tar.gz")
        truncate(p, 0.5)
        report = verify_report(tmp_path)
        assert not report.ok
        assert any("integrity failed" in x.detail for x in report.problems)

    def test_tar_gz_is_tar_checked_not_gzip_checked(self, tmp_path):
        """The double suffix must route to the tar verifier, which reads members;
        stopping at gzip would verify the envelope and not the contents."""
        assert _verifier_for(tmp_path / "bundle.tar.gz").__name__ == "_verify_tar"


# --------------------------------------------------------------------------
# honesty about what cannot be checked
# --------------------------------------------------------------------------
class TestUnverifiableFormats:
    def test_opaque_files_are_reported_not_silently_passed(self, tmp_path):
        (tmp_path / "weights.bin").write_bytes(b"\x00" * 1024)
        (tmp_path / "state.pt").write_bytes(b"\x00" * 1024)
        report = verify_report(tmp_path)
        assert report.ok                        # nothing detected wrong...
        assert len(report.unverifiable) == 2    # ...but nothing was vouched for
        assert report.verified == []
        assert "2 unverifiable" in report.render()

    def test_a_truncated_opaque_file_cannot_be_caught(self, tmp_path):
        """States the limit as a test rather than only in prose.

        A bare .bin carries no internal checksum, so truncation is undetectable
        here. If this ever starts failing, coverage improved and the module
        docstring's honesty clause should be revisited.
        """
        p = tmp_path / "weights.bin"
        p.write_bytes(b"\x00" * 1024)
        truncate(p)
        report = verify_report(tmp_path)
        assert report.ok
        assert p in report.unverifiable


# --------------------------------------------------------------------------
# residue and the empty-fetch trap
# --------------------------------------------------------------------------
class TestResidue:
    @pytest.mark.parametrize("suffix", [".tmp", ".part", ".partial"])
    def test_residue_warns_without_failing(self, leg_dir, suffix):
        (leg_dir / f"field.npz{suffix}").write_bytes(b"half a file")
        report = verify_report(leg_dir)
        assert report.ok, "residue alone must not fail an otherwise-good leg"
        assert any(not p.fatal for p in report.problems)

    def test_residue_is_not_itself_verified(self, leg_dir):
        """A .tmp is skipped, not checked -- it is expected to be garbage."""
        (leg_dir / "field.npz.tmp").write_bytes(b"not a zip")
        report = verify_report(leg_dir)
        assert report.ok
        assert not any(p.fatal for p in report.problems)

    def test_residue_alongside_real_damage_still_fails(self, leg_dir):
        (leg_dir / "field.npz.tmp").write_bytes(b"garbage")
        truncate(leg_dir / "field.npz")
        assert not verify_report(leg_dir).ok


class TestEmptyFetch:
    def test_empty_directory_warns_but_does_not_fail(self, tmp_path):
        """Reported, deliberately not fatal.

        `done_when` is the CALLER's declaration of completeness, and a leg whose
        marker is the fetch directory is legitimately satisfied by an empty one.
        Overriding that here would redefine a caller's contract in the name of
        integrity. The warning is the prompt to set a precise marker instead.
        """
        d = tmp_path / "out_leg"
        d.mkdir()
        report = verify_report(d)
        assert report.ok
        assert any("nothing was transferred" in p.detail and not p.fatal
                   for p in report.problems)

    def test_missing_directory_is_a_failure(self, tmp_path):
        assert not verify_report(tmp_path / "nope").ok

    def test_file_instead_of_directory_is_a_failure(self, tmp_path):
        p = tmp_path / "afile"
        p.write_text("x")
        assert not verify_report(p).ok

    def test_a_directory_of_only_residue_is_empty(self, tmp_path):
        """Residue is skipped, so a dir containing nothing else fetched nothing.

        Two warnings, no failure: the interrupted write AND the empty result.
        """
        d = tmp_path / "out_leg"
        d.mkdir()
        (d / "field.npz.tmp").write_bytes(b"partial")
        report = verify_report(d)
        assert report.ok
        assert len(report.problems) == 2 and not any(p.fatal for p in report.problems)


# --------------------------------------------------------------------------
# recursion + required artifacts + reporting
# --------------------------------------------------------------------------
class TestNestedLegDirectories:
    def _battery(self, tmp_path):
        root = tmp_path / "out_battery"
        for arm in ("neg", "pos"):
            d = root / arm
            d.mkdir(parents=True)
            make_zip(d / "field.npz")
        return root

    def test_walks_subdirectories(self, tmp_path):
        report = verify_report(self._battery(tmp_path))
        assert report.ok and len(report.verified) == 2

    def test_one_bad_arm_fails_the_fetch(self, tmp_path):
        root = self._battery(tmp_path)
        truncate(root / "pos" / "field.npz")
        report = verify_report(root)
        assert not report.ok
        assert any("pos" in str(p.path) for p in report.problems)


class TestRequiredArtifacts:
    def test_absent_required_file_is_fatal(self, leg_dir):
        report = verify_report(leg_dir, require=["summary.json"])
        assert not report.ok
        assert any(p.kind == "missing" for p in report.problems)

    def test_present_required_file_passes(self, leg_dir):
        assert verify_report(leg_dir, require=["manifest.json"]).ok


class TestReport:
    def test_summary_names_the_first_failure(self, leg_dir):
        truncate(leg_dir / "field.npz")
        assert "field.npz" in verify_report(leg_dir).summary()

    def test_summary_counts_additional_failures(self, tmp_path):
        d = tmp_path / "out_leg"
        d.mkdir()
        for name in ("a.npz", "b.npz"):
            truncate(make_zip(d / name))
        assert "+1 more" in verify_report(d).summary()

    def test_summary_carries_the_kind_for_a_leg_detail_field(self, leg_dir):
        """FleetExecutor puts this string in LegResult.detail, where the kind is
        the part that says what to do about it."""
        truncate(leg_dir / "field.npz")
        assert "[truncated]" in verify_report(leg_dir).summary()

    def test_empty_report_is_ok(self):
        assert ArrivalReport(root=None).ok

    def test_verify_tree_returns_the_same_problems(self, leg_dir):
        """`verify_tree` is the list-only view of `verify_report`; if they can
        disagree, one of them is lying."""
        from run_farm.arrival import verify_tree
        truncate(leg_dir / "field.npz")
        assert verify_tree(leg_dir) == verify_report(leg_dir).problems
