"""The launch gauntlet.

The load-bearing test in this file is
`test_missing_ssh_key_is_caught_before_any_rental`: that is the 7-rental, 72-minute
failure, and the reason an API call must never stand in for an SSH test.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from run_farm import gauntlet as gt
from run_farm.payload import PayloadSpec
from run_farm.gauntlet import (CheckResult, GauntletError, OffersAvailable,
                               OutDirWritable, PayloadClosed, ProviderCapable,
                               ResumeMarkersIntended, SshKeyPresent,
                               SshKeyRegistered, require_gauntlet, run_gauntlet,
                               standard_gauntlet)
from run_farm.protocols import HostSpec, Offer, RunHandle


def _keypair(tmp_path, name="vastai"):
    """A real ed25519 keypair, so fingerprints are real too."""
    key = tmp_path / name
    subprocess.run(["ssh-keygen", "-t", "ed25519", "-N", "", "-q", "-f", str(key)],
                   check=True, capture_output=True)
    return key


def _offer(i, dph=0.30):
    return Offer(id=i, dph=dph, gpu_name="RTX_3090", num_gpus=1, reliability=0.99,
                 inet_down_mbps=500.0, cuda_max=12.4)


class FakeProvider:
    name = "fake"

    def __init__(self, offers=(), keys=None):
        self._offers, self._keys = list(offers), keys

    def offers(self, spec):
        return list(self._offers)

    def rent(self, offer, launch, *, timeout_s=600):     # pragma: no cover
        raise AssertionError("the gauntlet must never rent")

    def destroy(self, host_id):                          # pragma: no cover
        pass

    def registered_ssh_keys(self):
        if self._keys is None:
            raise AssertionError("provider has no registered_ssh_keys")
        return list(self._keys)


# ------------------------------------------------------- the 7-rental failure
def test_missing_ssh_key_is_caught_before_any_rental(tmp_path):
    """THE test. `FleetExecutor` defaults to key_path=~/.ssh/vastai; that file did
    not exist; every host refused the connection and each refusal looked like a bad
    host. 7 rentals, 72 minutes, one absent local file."""
    check = SshKeyPresent(str(tmp_path / "does-not-exist"))
    r = check()
    assert not r.ok and r.blocking
    assert "does not exist" in r.detail
    assert "look like a bad host" in r.detail, "must name the misleading symptom"


def test_ssh_key_present_states_what_it_does_NOT_prove(tmp_path):
    """A green check that overstates itself is how the API-call-for-SSH-test
    substitution happened in the first place."""
    r = SshKeyPresent(str(_keypair(tmp_path)))()
    assert r.ok
    assert "Does NOT prove" in r.proves and "reachable" in r.proves


def test_group_readable_key_is_rejected(tmp_path):
    key = _keypair(tmp_path)
    key.chmod(0o644)
    r = SshKeyPresent(str(key))()
    assert not r.ok and "mode" in r.detail and "chmod 600" in r.detail


def test_missing_public_half_is_rejected(tmp_path):
    key = _keypair(tmp_path)
    key.with_suffix(".pub").unlink()
    r = SshKeyPresent(str(key))()
    assert not r.ok and "fingerprint cannot be compared" in r.detail


# --------------------------------------------------------------- registration
def test_registered_key_passes_and_unregistered_key_fails(tmp_path):
    key = _keypair(tmp_path)
    mine = key.with_suffix(".pub").read_text()
    other = _keypair(tmp_path, "other").with_suffix(".pub").read_text()

    ok = SshKeyRegistered(FakeProvider(keys=[other, mine]), str(key))()
    assert ok.ok and "registered" in ok.detail

    bad = SshKeyRegistered(FakeProvider(keys=[other]), str(key))()
    assert not bad.ok and "NOT among" in bad.detail       # the negative control


def test_unverifiable_registration_FAILS_rather_than_passing(tmp_path):
    """'I could not look' must not render as 'it is fine' -- the whole bug class."""
    class NoKeyListing:
        name = "nokeys"

    r = SshKeyRegistered(NoKeyListing(), str(_keypair(tmp_path)))()
    assert not r.ok, "an unverifiable check must not pass"
    assert "CANNOT be verified" in r.detail and "must not read as a pass" in r.detail


# -------------------------------------------------------------------- payload
def test_payload_check_delegates_and_reports_problems(tmp_path):
    (tmp_path / "a.py").write_text("X = 1\n")
    good = PayloadClosed(PayloadSpec(files=(tmp_path / "a.py",), imports=("a",)))()
    assert good.ok

    bad = PayloadClosed(PayloadSpec(files=(tmp_path / "gone.py",)))()
    assert not bad.ok and "missing_file" in bad.detail


# --------------------------------------------------------------- marketplace
def test_offers_available_uses_a_free_call_and_fails_on_an_empty_pool():
    ok = OffersAvailable(FakeProvider([_offer("a"), _offer("b")]), HostSpec())()
    assert ok.ok and "2 offer(s)" in ok.detail

    empty = OffersAvailable(FakeProvider([]), HostSpec())()
    assert not empty.ok and "relax" in empty.detail
    assert "Does NOT reserve" in empty.proves, "must not imply the offers are held"


# --------------------------------------------------------------------- local
def test_out_dir_writable_and_its_negative_control(tmp_path):
    assert OutDirWritable(tmp_path / "out")().ok

    locked = tmp_path / "locked"
    locked.mkdir(mode=0o500)
    try:
        r = OutDirWritable(locked / "sub")()
        assert not r.ok
    finally:
        locked.chmod(0o700)


# ------------------------------------------------------- the skip-that-passed
class _Leg:
    def __init__(self, label, marker):
        self.label, self._marker = label, marker

    def marker(self):
        return self._marker


def test_pre_skipping_legs_are_reported_loudly(tmp_path):
    """A leg reported SKIP (output already present) and that read as a pass -- the
    marker had been left by a DIFFERENT PROVIDER's earlier run. A skip is a claim
    that work is done; it deserves the same scrutiny as a result."""
    (tmp_path / "L1").mkdir(parents=True)
    (tmp_path / "L1" / "manifest.json").write_text("{}")
    legs = [_Leg("L1", "manifest.json"), _Leg("L2", "manifest.json")]

    r = ResumeMarkersIntended(legs, tmp_path)()
    assert not r.ok and not r.fatal, "loud, but a human may legitimately accept it"
    assert "will be SKIPPED" in r.detail and "L1" in r.detail
    assert "THIS campaign and provider" in r.detail
    assert "h old" in r.detail, "marker age makes a stale marker visible"


def test_no_markers_means_nothing_pre_skips(tmp_path):
    r = ResumeMarkersIntended([_Leg("L1", "m.json")], tmp_path)()
    assert r.ok and "no leg pre-skips" in r.detail


def test_a_leg_with_no_marker_can_never_resume_and_is_flagged(tmp_path):
    r = ResumeMarkersIntended([_Leg("L1", "")], tmp_path)()
    assert not r.ok and "can never resume" in r.detail


# ---------------------------------------------------------------- capability
def test_missing_required_method_blocks_launch():
    """The broken-monitor failure, moved to launch time."""
    class NoDestroy:
        name = "nodestroy"

        def offers(self, spec): return []
        def rent(self, o, l, *, timeout_s=600): pass

    r = ProviderCapable(NoDestroy(), required=("offers", "rent", "destroy"))()
    assert not r.ok and "MISSING required" in r.detail and "destroy" in r.detail
    assert "AttributeError" in r.detail


def test_absent_optional_method_is_a_pass_but_names_the_degradation():
    r = ProviderCapable(FakeProvider(), required=("offers", "rent", "destroy"),
                        optional=("logs",))()
    assert r.ok, "a thin provider is usable"
    assert "degraded" in r.detail and "logs" in r.detail


# ------------------------------------------------------------------ gauntlet
def test_gauntlet_reports_every_failure_not_just_the_first(tmp_path):
    results = run_gauntlet([SshKeyPresent(str(tmp_path / "nope")),
                        PayloadClosed(PayloadSpec(files=(tmp_path / "gone.py",))),
                        OffersAvailable(FakeProvider([]), HostSpec())], log=None)
    assert len(results) == 3 and all(not r.ok for r in results)


def test_a_check_that_RAISES_is_reported_as_a_failure(tmp_path):
    """A broken check must never read as a quiet pass -- that is precisely how a
    monitor calling a nonexistent method looked healthy."""
    def exploding():
        raise RuntimeError("check is broken")
    exploding.name = "exploding"

    [r] = run_gauntlet([exploding], log=None)
    assert not r.ok and "the CHECK ITSELF raised" in r.detail


def test_require_raises_with_all_blockers_and_ignores_warnings(tmp_path):
    key = _keypair(tmp_path)
    (tmp_path / "L1").mkdir()
    (tmp_path / "L1" / "m.json").write_text("{}")

    with pytest.raises(GauntletError) as ei:
        require_gauntlet([SshKeyPresent(str(tmp_path / "nope")),
                 ResumeMarkersIntended([_Leg("L1", "m.json")], tmp_path)], log=None)
    assert len(ei.value.blocking) == 1, "the non-fatal warning must not block"
    assert len(ei.value.results) == 2, "but it is still reported"

    # negative control: all-passing checks do not raise
    require_gauntlet([SshKeyPresent(str(key)), OutDirWritable(tmp_path / "o")], log=None)


def test_standard_gauntlet_is_ordered_cheapest_first_and_honours_skip(tmp_path):
    key = _keypair(tmp_path)
    checks = standard_gauntlet(provider=FakeProvider([_offer("a")], keys=[]),
                               host_spec=HostSpec(), out_dir=tmp_path,
                               key_path=str(key))
    names = [c.name for c in checks]
    assert names.index("ssh-key-present") < names.index("offers-available"), \
        "a local file check must precede a network round trip"

    skipped = standard_gauntlet(provider=FakeProvider([_offer("a")]),
                                host_spec=HostSpec(), out_dir=tmp_path,
                                key_path=str(key), skip=("ssh-key-registered",))
    assert "ssh-key-registered" not in [c.name for c in skipped]


def test_standard_gauntlet_end_to_end_passes_on_a_sound_setup(tmp_path):
    """The whole gauntlet green, with a real key, a real payload and real offers --
    so the failing cases above are demonstrably not failing for an unrelated reason."""
    key = _keypair(tmp_path)
    (tmp_path / "entry.py").write_text("VALUE = 1\n")
    prov = FakeProvider([_offer("a")], keys=[key.with_suffix(".pub").read_text()])

    results = require_gauntlet(standard_gauntlet(
        provider=prov, host_spec=HostSpec(), out_dir=tmp_path / "out",
        key_path=str(key),
        payload=PayloadSpec(files=(tmp_path / "entry.py",), imports=("entry",)),
    ), log=None)
    assert all(r.ok for r in results) and len(results) >= 5
    assert all(r.proves for r in results), "every check must state what it proves"


# ------------------------------------------------------- remote-env pinned ----
class _Exec:
    def __init__(self, remote_env=None):
        self.remote_env = dict(remote_env or {})


_XLA = {"XLA_FLAGS": "--xla_gpu_autotune_level=0"}


def test_remote_env_pinned_passes_when_set():
    r = gt.RemoteEnvPinned(_Exec(_XLA), _XLA)()
    assert r.ok and not r.blocking
    assert "XLA_FLAGS" in r.detail and r.proves


def test_remote_env_pinned_fails_when_missing():
    """The silent failure this guards: every leg runs and only the claim is void."""
    r = gt.RemoteEnvPinned(_Exec(), _XLA)()
    assert not r.ok and r.blocking          # fatal by default
    assert "missing: XLA_FLAGS" in r.detail


def test_remote_env_pinned_fails_on_a_wrong_value():
    """Set-but-wrong is the nastier case: it looks configured."""
    r = gt.RemoteEnvPinned(_Exec({"XLA_FLAGS": "--xla_gpu_autotune_level=4"}), _XLA)()
    assert not r.ok
    assert "wrong: XLA_FLAGS=" in r.detail


def test_remote_env_pinned_handles_an_executor_without_the_attribute():
    """An older executor has no remote_env; that must read as MISSING, not crash --
    a check that raises is reported as a failure, but a clear one is better."""
    r = gt.RemoteEnvPinned(object(), _XLA)()
    assert not r.ok and "missing: XLA_FLAGS" in r.detail


# -- cap vs worst case ---------------------------------------------------------
def test_cap_clears_worst_case_passes(tmp_path):
    from run_farm.gauntlet import CapClearsWorstCase
    r = CapClearsWorstCase(30.0, 25.62)()
    assert r.ok and "25.62" in r.detail


def test_cap_below_worst_case_warns_with_the_remedy():
    """The real 2026-08-06 parameterisation: a $38 cap against a $45.75 worst
    case. A cap under the worst case does not prevent overspend, it guarantees the
    campaign dies partway through having already paid."""
    from run_farm.gauntlet import CapClearsWorstCase
    r = CapClearsWorstCase(38.0, 45.75)()
    assert not r.ok
    assert "abort mid-campaign" in r.detail
    assert "45.75" in r.detail                     # names the number to clear
    assert r.fatal is False                        # a tight cap can be deliberate


def test_cap_check_counts_spend_already_on_the_ledger():
    """A ledger that has already absorbed spend leaves less headroom, and the cap
    is enforced against the running total, not against this campaign alone."""
    from run_farm.gauntlet import CapClearsWorstCase
    assert CapClearsWorstCase(26.0, 5.0, already_spent_usd=19.46)().ok
    r = CapClearsWorstCase(20.0, 5.0, already_spent_usd=19.46)()
    assert not r.ok and "19.46" in r.detail


# ------------------------------------------------------- registry path ----
#
# The load-bearing test in this section is
# `test_a_registry_skip_is_reported_rather_than_silent`: it is the same failure
# `ResumeMarkersIntended` exists for -- a skip reading as a pass -- on the path
# that had no check for it.


class _Cfg:
    """Minimal `RunConfig`: the registry path needs only identity from it."""

    def __init__(self, tag):
        self.tag = tag
        self.dtype = "float32"
        self.params = {}

    def to_json(self):
        return json.dumps({"tag": self.tag}, sort_keys=True)

    def config_hash(self, n=12):
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:n]

    def run_name(self):
        return f"run_{self.tag}_{self.config_hash()}"


class _Registry:
    """`RunRegistry` stub. Records whether `register` was ever called, because
    the check promising to be read-only is only worth having if that is tested."""

    def __init__(self, complete=(), raises=False):
        self.complete = set(complete)
        self.raises = raises
        self.registered = []

    def register(self, config):
        self.registered.append(config)
        return RunHandle(config=config, dir=Path("/nonexistent"),
                         name=config.run_name())

    def is_complete(self, handle):
        if self.raises:
            raise RuntimeError("registry unreachable")
        return handle.name in self.complete


def test_no_registry_skip_passes_and_says_how_many_will_run(tmp_path):
    cfgs = [_Cfg("a"), _Cfg("b")]
    r = _Registry()
    res = gt.RegistryMarkersIntended(r, cfgs, tmp_path)()
    assert res.ok and "all 2 will run" in res.detail
    assert res.proves


def test_a_registry_skip_is_reported_rather_than_silent(tmp_path):
    """THE one. A skip is a claim that work is already done; it must be named."""
    cfgs = [_Cfg("a"), _Cfg("b")]
    r = _Registry(complete={cfgs[1].run_name()})
    res = gt.RegistryMarkersIntended(r, cfgs, tmp_path)()
    assert not res.ok
    assert not res.fatal          # loud, but a human may legitimately accept it
    assert cfgs[1].run_name() in res.detail
    assert "SKIPPED" in res.detail
    assert cfgs[0].run_name() not in res.detail


def test_the_check_never_registers_anything(tmp_path):
    """Read-only is load-bearing: `run_campaign` defers registration to the
    worker on purpose, and a check that pre-registered would both reintroduce
    that cost and create run dirs for work that never happens."""
    cfgs = [_Cfg("a"), _Cfg("b")]
    r = _Registry(complete={cfgs[0].run_name()})
    gt.RegistryMarkersIntended(r, cfgs, tmp_path)()
    assert r.registered == []
    assert list(tmp_path.iterdir()) == []


def test_marker_age_is_reported_when_the_run_dir_is_real(tmp_path):
    cfg = _Cfg("a")
    (tmp_path / cfg.run_name()).mkdir()
    r = _Registry(complete={cfg.run_name()})
    res = gt.RegistryMarkersIntended(r, [cfg], tmp_path)()
    assert not res.ok and "h old" in res.detail


def test_a_registry_without_directories_still_reports_the_skip(tmp_path):
    """Age is best-effort; the skip itself is not. An object-store registry has
    no mtime to read and must still name what it will skip."""
    cfg = _Cfg("a")
    r = _Registry(complete={cfg.run_name()})
    res = gt.RegistryMarkersIntended(r, [cfg], tmp_path / "absent")()
    assert not res.ok and cfg.run_name() in res.detail and "h old" not in res.detail


def test_an_unreachable_registry_fails_fatally(tmp_path):
    """`is_complete` raising means the skip state is UNKNOWN, which is not the
    same as 'nothing will skip' and must not be reported as a pass."""
    res = gt.RegistryMarkersIntended(_Registry(raises=True), [_Cfg("a")], tmp_path)()
    assert not res.ok and res.fatal and "unreachable" in res.detail


def test_limit_says_what_it_did_not_examine(tmp_path):
    cfgs = [_Cfg(str(i)) for i in range(5)]
    res = gt.RegistryMarkersIntended(_Registry(), cfgs, tmp_path, limit=2)()
    assert res.ok and "3 more not examined" in res.detail


def test_runfn_importable_passes_on_a_real_run_fn():
    """Pointed at an actual shipped `RunFn`, not a maybe-present one: a check
    whose positive case can skip itself proves nothing about the positive case."""
    res = gt.RunFnImportable("run_farm.testing:echo_run_fn")()
    assert res.ok, res.detail
    assert "resolves" in res.detail


@pytest.mark.parametrize("ref", ["run_farm.nope:missing",
                                 "run_farm.gauntlet:no_such_function"])
def test_a_stale_runfn_reference_fails_before_launch(ref):
    """Otherwise this fails identically on every leg, at rental prices."""
    res = gt.RunFnImportable(ref)()
    assert not res.ok and res.fatal
    assert res.proves and "Does NOT prove" in res.proves


def test_a_non_callable_runfn_reference_fails():
    """`load_run_fn` raises TypeError rather than returning a non-callable, and
    the check must surface that as a failure and not an exception."""
    res = gt.RunFnImportable("run_farm.gauntlet:__doc__")()
    assert not res.ok and res.fatal


def test_registry_gauntlet_covers_the_path_and_skips_by_name(tmp_path):
    cfgs = [_Cfg("a")]
    checks = gt.registry_gauntlet(registry=_Registry(), configs=cfgs,
                                  out_dir=tmp_path,
                                  run_fn_ref="run_farm.gauntlet:require_gauntlet")
    names = [c.name for c in checks]
    assert names == ["out-dir-writable", "runfn-importable",
                     "registry-markers-intended"]
    dropped = gt.registry_gauntlet(registry=_Registry(), configs=cfgs,
                                   out_dir=tmp_path,
                                   skip=("registry-markers-intended",))
    assert [c.name for c in dropped] == ["out-dir-writable"]


def test_registry_gauntlet_runs_green_end_to_end(tmp_path):
    checks = gt.registry_gauntlet(registry=_Registry(), configs=[_Cfg("a")],
                                  out_dir=tmp_path,
                                  run_fn_ref="run_farm.gauntlet:require_gauntlet")
    results = gt.run_gauntlet(checks)
    assert all(r.ok for r in results), [str(r) for r in results]
    assert all(r.proves for r in results)


class _DirKeyedRegistry:
    """Mimics `FileRunRegistry` exactly: completion resolves through `handle.dir`.

    The point of the class is the contrast with `_Registry` above, which keys off
    `handle.name` the way `ObjectStoreRunRegistry` does. The two disagree, and a
    handle built under the wrong directory is invisible to only one of them.
    """

    def __init__(self, base):
        self.base = Path(base)

    def register(self, config):
        d = self.base / config.run_name()
        d.mkdir(parents=True, exist_ok=True)
        return RunHandle(config=config, dir=d, name=config.run_name())

    def finish(self, handle, result):
        handle.dir.mkdir(parents=True, exist_ok=True)
        (handle.dir / "DONE.json").write_text(json.dumps(result))

    def is_complete(self, handle):
        return (handle.dir / "DONE.json").exists()


def test_a_dir_keyed_registry_reports_its_skip_when_the_base_matches(tmp_path):
    base = tmp_path / "campaign_out"
    reg = _DirKeyedRegistry(base)
    cfgs = [_Cfg("a"), _Cfg("b")]
    reg.finish(reg.register(cfgs[1]), {"ok": True})
    res = gt.RegistryMarkersIntended(reg, cfgs, base)()
    assert not res.ok and not res.fatal
    assert cfgs[1].run_name() in res.detail


def test_a_base_mismatch_fails_instead_of_passing_cleanly(tmp_path):
    """REGRESSION. `FileRunRegistry.is_complete` reads `handle.dir`, so a handle
    built under a different directory makes every finished run report unfinished
    and this check returned a clean pass — a false all-clear on precisely the
    failure it exists to catch. Worse than the unreachable-registry case, which is
    already fatal: unknown skip state is loud, wrong skip state is silent.
    """
    base = tmp_path / "campaign_out"
    reg = _DirKeyedRegistry(base)
    cfgs = [_Cfg("a"), _Cfg("b")]
    reg.finish(reg.register(cfgs[1]), {"ok": True})
    assert reg.is_complete(reg.register(cfgs[1]))     # the run really is complete

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    res = gt.RegistryMarkersIntended(reg, cfgs, elsewhere)()
    assert not res.ok and res.fatal, "a base mismatch must not read as a pass"
    assert "not the registry's base" in res.detail


def test_a_name_keyed_registry_is_unaffected_by_the_directory(tmp_path):
    """The asymmetry that made the bug easy to miss: an object-store registry
    keys off `handle.name`, so it never had the problem. `_Registry` exposes no
    `base`, so the guard stays quiet rather than inventing a mismatch."""
    cfgs = [_Cfg("a")]
    r = _Registry(complete={cfgs[0].run_name()})
    res = gt.RegistryMarkersIntended(r, cfgs, tmp_path / "anywhere")()
    assert not res.ok and not res.fatal
    assert cfgs[0].run_name() in res.detail


def test_handle_for_overrides_the_layout_and_suppresses_the_base_guard(tmp_path):
    base = tmp_path / "campaign_out"
    reg = _DirKeyedRegistry(base)
    cfgs = [_Cfg("a")]
    reg.finish(reg.register(cfgs[0]), {"ok": True})
    res = gt.RegistryMarkersIntended(
        reg, cfgs, tmp_path / "irrelevant",
        handle_for=lambda c: RunHandle(config=c, dir=base / c.run_name(),
                                       name=c.run_name()))()
    assert not res.ok and not res.fatal
    assert cfgs[0].run_name() in res.detail


def test_a_zero_limit_fails_rather_than_passing_with_no_coverage(tmp_path):
    """A check that examines nothing cannot fail, and this module's rule is that
    something which cannot fail is not a check."""
    res = gt.RegistryMarkersIntended(_Registry(), [_Cfg("a")], tmp_path, limit=0)()
    assert not res.ok and res.fatal and "cannot fail" in res.detail
