"""FleetExecutor tests: a FakeProvider (no spend) + monkeypatched ssh/scp cover
the parallel one-host-per-leg script fleet -- happy run, per-leg failover on a
bad host / offer race, fast-fail via API status (#27), offer-pool refresh (#28),
resume/skip (#26), the NO_OFFERS / RUN_FAIL / NO_RESULT outcomes, and the
signal-safe teardown backstop (#24).
"""

import contextlib
import signal
import threading
import types

import pytest

import run_farm.fleet as fleet
from run_farm import (FleetExecutor, FleetLeg, HostProbeFailed,
                                   HostSpec, LaunchSpec, LegResult, Offer,
                                   RentedHost, RentUnavailable, SentinelReady,
                                   fleet_status)

LAUNCH = LaunchSpec(image="img:12.2", onstart="echo hi", disk_gb=24, label="farm-x")
SPEC = HostSpec(gpu_name="RTX 3090", max_dph=0.30)


def _offer(oid, dph=0.12):
    return Offer(id=oid, dph=dph, gpu_name="RTX 3090", num_gpus=1, reliability=0.99,
                 inet_down_mbps=800, cuda_max=12.4, geolocation="x", provider="fake")


def _leg(label, *, fetch="", done_when="", command="run.sh"):
    return FleetLeg(label=label, command=command, ship=("driver.py",),
                    fetch=fetch, done_when=done_when)


class FakeProvider:
    """offers() (with one refill) + leak-proof rent(); bad_ids fail to come up,
    race_ids are taken (RentUnavailable), dead_ids never get ready but report
    dead via the API (the #27 fast-fail path). destroy/list_instances/dead_reason
    round out the surface FleetExecutor + status use."""

    name = "fake"

    def __init__(self, offers, *, bad_ids=(), race_ids=(), dead_ids=(), refill=None):
        self._offers = list(offers)
        self._bad, self._race, self._dead = set(bad_ids), set(race_ids), set(dead_ids)
        self._refill = refill
        self.offers_calls = 0
        self.live: dict[str, Offer] = {}
        self.rented: list[str] = []
        self.destroyed: list[str] = []
        self._n = 0
        self._lock = threading.Lock()

    def offers(self, spec):
        with self._lock:
            self.offers_calls += 1
            return list(self._offers if self.offers_calls == 1 else (self._refill or []))

    @contextlib.contextmanager
    def rent(self, offer, launch, *, timeout_s=600):
        if offer.id in self._race:                        # taken between offers/rent
            raise RentUnavailable(f"offer {offer.id} taken")
        with self._lock:
            iid = str(1000 + self._n); self._n += 1
            self.rented.append(offer.id); self.live[iid] = offer
        try:
            if offer.id in self._bad:
                raise HostProbeFailed(f"bad host {offer.id}")
            yield RentedHost(id=iid, ssh_host="10.0.0.1", ssh_port=22, offer=offer)
        finally:
            with self._lock:
                self.live.pop(iid, None)                  # leak-proof teardown

    def dead_reason(self, instance_id):
        offer = self.live.get(str(instance_id))
        if offer is not None and offer.id in self._dead:
            return f"container error on {offer.id}"
        return None

    def destroy(self, instance_id):
        with self._lock:
            self.destroyed.append(str(instance_id))
            self.live.pop(str(instance_id), None)

    def list_instances(self):
        with self._lock:
            return [types.SimpleNamespace(id=int(i), status="running", dph=o.dph)
                    for i, o in self.live.items()]


def _fake_ssh_factory(*, ready=True, run_rc=0, run_out="done"):
    """Route the readiness probe (import / sentinel ls) vs the leg command."""
    def fake_ssh(key, host, port, cmd, timeout=120):
        if "import run_farm" in cmd:
            return (0, "") if ready else (1, "ModuleNotFoundError")
        if "worker-ready" in cmd:                          # SentinelReady probe
            return (0, "/tmp/worker-ready") if ready else (0, "")
        return (run_rc, run_out)                            # the leg command
    return fake_ssh


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(fleet.time, "sleep", lambda s: None)
    monkeypatch.setattr(fleet, "_scp_up", lambda *a, **k: (0, ""))
    monkeypatch.setattr(fleet, "_scp_down", lambda *a, **k: (0, ""))

    def apply(**kw):
        monkeypatch.setattr(fleet, "_ssh", _fake_ssh_factory(**kw))
    return apply


def _exec(prov, tmp_path, **kw):
    kw.setdefault("ready_timeout", 0.2)                     # short busy-wait on timeout
    return FleetExecutor(prov, LAUNCH, local_out_dir=str(tmp_path),
                         host_spec=SPEC, jitter_s=0, ready_poll_s=0,
                         log=lambda *_a: None, **kw)


def test_legresult_host_id_is_instance_not_offer(patched, tmp_path):
    """host_id correlates to the rented INSTANCE (host.id), not the offer id, so
    a result can be matched against the provider's live-instance list."""
    patched(ready=True)
    prov = FakeProvider([_offer("offer-a")])
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "OK"
    assert r.host_id == "1000" and r.host_id != "offer-a"   # instance id, not offer


def test_happy_path_runs_all_legs_and_tears_down(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("a"), _offer("b")])
    legs = [_leg("L1"), _leg("L2")]
    results = _exec(prov, tmp_path).run(legs)
    assert {r.status for r in results} == {"OK"}
    assert len(prov.rented) == 2 and prov.live == {}        # both torn down
    assert [r.label for r in results] == ["L1", "L2"]       # order preserved


def test_resume_skips_already_complete_leg(patched, tmp_path):
    patched(ready=True)
    # pre-create L1's output marker -> it should be SKIPped, only L2 runs
    (tmp_path / "L1").mkdir()
    (tmp_path / "L1" / "manifest.json").write_text("{}")
    legs = [_leg("L1", fetch="out", done_when="manifest.json"),
            _leg("L2", fetch="out", done_when="manifest.json")]
    prov = FakeProvider([_offer("a")])
    # L2's marker won't exist after the no-op fetch -> NO_RESULT (proves it ran)
    results = {r.label: r for r in _exec(prov, tmp_path).run(legs)}
    assert results["L1"].status == "SKIP"
    assert results["L2"].status == "NO_RESULT"
    assert prov.rented == ["a"]                             # only L2 rented a box


def test_fails_over_bad_host(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("bad"), _offer("good")], bad_ids={"bad"})
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "OK" and prov.rented == ["bad", "good"]
    assert prov.live == {}                                   # both attempts torn down


def test_fails_over_offer_race(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("taken"), _offer("free")], race_ids={"taken"})
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "OK" and prov.rented == ["free"]      # raced offer never rented


def test_fast_fail_dead_host_via_api_status(patched, tmp_path):
    """A host that comes up SSH-reachable but whose container died never passes
    the ready probe; dead_reason fast-fails it to the next offer (#27). Here both
    hosts stay 'not ready': 'dead' fast-fails at once via the API, 'good' only
    times out -- both get tried, then the pool exhausts."""
    patched(ready=False)                                    # ready probe never passes
    prov = FakeProvider([_offer("dead"), _offer("good")], dead_ids={"dead"})
    [r] = _exec(prov, tmp_path, max_refills=0).run([_leg("L1")])
    assert prov.rented == ["dead", "good"]
    assert r.status == "NO_OFFERS"
    assert prov.live == {}


def test_offer_pool_refreshes_when_drained(patched, tmp_path):
    """First offer is bad and drains the pool; a refill query supplies a good
    one so the leg still completes (#28)."""
    patched(ready=True)
    prov = FakeProvider([_offer("bad")], bad_ids={"bad"}, refill=[_offer("good")])
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "OK" and prov.rented == ["bad", "good"]
    assert prov.offers_calls >= 2                            # re-queried after drain


def test_no_offers_when_pool_exhausts(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("bad")], bad_ids={"bad"})    # no refill
    [r] = _exec(prov, tmp_path, max_refills=0).run([_leg("L1")])
    assert r.status == "NO_OFFERS"
    assert prov.live == {}


def test_run_fail_on_nonzero_command(patched, tmp_path):
    patched(ready=True, run_rc=1, run_out="boom")
    prov = FakeProvider([_offer("a")])
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "RUN_FAIL" and "boom" in r.detail
    assert prov.live == {}                                   # torn down anyway


def test_no_result_when_marker_absent(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    [r] = _exec(prov, tmp_path).run([_leg("L1", fetch="out", done_when="manifest.json")])
    assert r.status == "NO_RESULT"                           # ran ok but no marker


def test_duplicate_labels_rejected(patched, tmp_path):
    patched(ready=True)
    ex = _exec(FakeProvider([_offer("a")]), tmp_path)
    with pytest.raises(ValueError, match="unique"):
        ex.run([_leg("dup"), _leg("dup")])


def test_empty_legs_is_noop(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    assert _exec(prov, tmp_path).run([]) == []
    assert prov.rented == []


def test_sentinel_ready_probe(patched, tmp_path):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path, ready=SentinelReady())
    [r] = ex.run([_leg("L1")])
    assert r.status == "OK"


def test_sentinel_bad_network_fails_over(monkeypatch, tmp_path):
    """The onstart net-probe sentinel makes a throttled host fail over at once."""
    monkeypatch.setattr(fleet.time, "sleep", lambda s: None)
    monkeypatch.setattr(fleet, "_scp_up", lambda *a, **k: (0, ""))
    monkeypatch.setattr(fleet, "_scp_down", lambda *a, **k: (0, ""))

    def ssh(key, host, port, cmd, timeout=120):
        if "worker-ready" in cmd:                           # the SentinelReady ls
            return (0, "/tmp/worker-bad-network")           # only the BAD sentinel
        return (0, "done")
    monkeypatch.setattr(fleet, "_ssh", ssh)
    # every host reports bad-network -> each fails over -> pool exhausts
    prov = FakeProvider([_offer("bad"), _offer("good")])
    [r] = _exec(prov, tmp_path, ready=SentinelReady(), max_refills=0).run([_leg("L1")])
    assert r.status == "NO_OFFERS"
    assert prov.rented == ["bad", "good"]


def test_destroy_live_backstop(tmp_path):
    """The signal backstop force-destroys every tracked rental."""
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path)
    ex._track("1000"); ex._track("1001")
    ex._destroy_live()
    assert set(prov.destroyed) == {"1000", "1001"}


def test_signal_guard_installs_and_restores(tmp_path):
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path)
    before = signal.getsignal(signal.SIGTERM)
    with ex._signal_guard():
        assert signal.getsignal(signal.SIGTERM) not in (before, None)  # installed
    assert signal.getsignal(signal.SIGTERM) is before                  # restored


def test_signal_handler_destroys_then_chains(tmp_path):
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path)
    ex._track("1000")
    chained = []
    guard = fleet._SignalGuard(ex)
    guard._prev = {signal.SIGTERM: lambda *_a: chained.append(True)}
    guard._handle(signal.SIGTERM, None)
    assert prov.destroyed == ["1000"] and chained == [True]


def test_fleet_status_reports_live_and_spend(tmp_path):
    from run_farm.vast import VastLedger
    prov = FakeProvider([_offer("a")])
    prov.live["1000"] = _offer("a", dph=0.2)
    led = VastLedger(tmp_path / "l.jsonl")
    led.record("destroyed", outcome="ok", billed_s=120, est_cost_usd=0.05)
    snap = fleet_status(prov, led)
    assert snap["live_dph"] == 0.2 and len(snap["live"]) == 1
    assert snap["ledger"]["total_est_cost_usd"] == 0.05
    assert snap["ledger"]["by_outcome"] == {"ok": 1}


def test_launch_jitter_staggers_starts(monkeypatch, tmp_path):
    """With jitter on, the i-th leg of a wave waits i*jitter before renting (#29)."""
    sleeps = []
    monkeypatch.setattr(fleet.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(fleet, "_scp_up", lambda *a, **k: (0, ""))
    monkeypatch.setattr(fleet, "_scp_down", lambda *a, **k: (0, ""))
    monkeypatch.setattr(fleet, "_ssh", _fake_ssh_factory(ready=True))
    ex = FleetExecutor(FakeProvider([_offer("a"), _offer("b")]), LAUNCH,
                       local_out_dir=str(tmp_path), host_spec=SPEC, jitter_s=0.01,
                       ready_timeout=0.2, ready_poll_s=0, max_parallel=4,
                       log=lambda *_a: None)
    ex.run([_leg("L0"), _leg("L1")])
    assert any(s > 0 for s in sleeps)                      # at least one staggered


def test_legresult_ok_flag():
    assert LegResult("x", "OK").ok and LegResult("x", "SKIP").ok
    assert not LegResult("x", "RUN_FAIL").ok


def test_ship_failure_fails_over(monkeypatch, tmp_path):
    """A failed scp-up of the driver is a host problem -> fail over (#25)."""
    monkeypatch.setattr(fleet.time, "sleep", lambda s: None)
    monkeypatch.setattr(fleet, "_scp_down", lambda *a, **k: (0, ""))
    monkeypatch.setattr(fleet, "_scp_up", lambda *a, **k: (1, "scp boom"))
    monkeypatch.setattr(fleet, "_ssh", _fake_ssh_factory(ready=True))
    prov = FakeProvider([_offer("a")])
    [r] = _exec(prov, tmp_path, max_refills=0).run([_leg("L1")])
    assert r.status == "NO_OFFERS" and prov.live == {}     # ship failed -> drained


class _RaiseProvider(FakeProvider):
    """rent() raises a non-failover error on __enter__ (terminal)."""

    def __init__(self, offers, msg):
        super().__init__(offers)
        self._msg = msg

    @contextlib.contextmanager
    def rent(self, offer, launch, *, timeout_s=600):
        self.rented.append(offer.id)
        raise RuntimeError(self._msg)
        yield  # pragma: no cover  (unreachable; makes this a generator cm)


def test_generic_error_is_terminal(patched, tmp_path):
    patched(ready=True)
    prov = _RaiseProvider([_offer("a")], "kaboom")
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "ERROR" and "kaboom" in r.detail


def test_leak_surfaces_as_leak_status(patched, tmp_path):
    patched(ready=True)
    prov = _RaiseProvider([_offer("a")], "LEAK RISK: instance 9 not torn down")
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "LEAK"


def test_destroy_live_no_destroy_method_is_noop(tmp_path):
    class NoDestroy:
        name = "nd"
        def offers(self, spec):
            return []
    ex = _exec(NoDestroy(), tmp_path)
    ex._track("1")
    ex._destroy_live()                                     # no destroy attr -> no-op


def test_destroy_live_logs_on_failure(tmp_path):
    msgs = []

    class FailDestroy(FakeProvider):
        def destroy(self, iid):
            raise RuntimeError("nope")
    ex = FleetExecutor(FailDestroy([_offer("a")]), LAUNCH,
                       local_out_dir=str(tmp_path), log=msgs.append)
    ex._track("1000")
    ex._destroy_live()
    assert any("FAILED to destroy" in m for m in msgs)


def test_signal_handle_default_raises_keyboardinterrupt(tmp_path):
    ex = _exec(FakeProvider([_offer("a")]), tmp_path)
    guard = fleet._SignalGuard(ex)
    guard._prev = {signal.SIGTERM: signal.SIG_DFL}         # not callable -> default
    with pytest.raises(KeyboardInterrupt):
        guard._handle(signal.SIGTERM, None)


def test_signal_guard_exit_skips_none_prev(tmp_path):
    ex = _exec(FakeProvider([_offer("a")]), tmp_path)
    guard = fleet._SignalGuard(ex)
    guard._prev = {signal.SIGTERM: None}                   # was set from C
    guard.__exit__()                                       # must not raise


def test_fleet_status_single_source(tmp_path):
    from run_farm.vast import VastLedger
    prov = FakeProvider([_offer("a")])
    prov.live["1"] = _offer("a", dph=0.1)
    assert "ledger" not in fleet_status(prov, None)
    led = VastLedger(tmp_path / "l.jsonl")
    assert "live" not in fleet_status(None, led)


def test_status_format_variants():
    from run_farm.status import _format
    assert "no provider" in _format({})
    s = _format({"live": [], "live_dph": 0.0,
                 "ledger": {"rentals": 0, "total_billed_min": 0,
                            "total_est_cost_usd": 0.0, "by_outcome": {}}})
    assert "none" in s and "LEDGER" in s
    s2 = _format({"live": [{"id": 5, "status": "running", "dph": 0.2}],
                  "live_dph": 0.2})
    assert "0.200/hr" in s2


# -- in-flight host-death + scp/marker + post-run sweep (CM review #48) --------
def test_inflight_retry_on_confirmed_host_death(patched, tmp_path, monkeypatch):
    """A leg whose command exits non-zero on a host that `dead_reason` CONFIRMS is
    dead fails over and re-runs on a fresh host -- not a terminal RUN_FAIL."""
    leg_calls = {"n": 0}

    def stateful_ssh(key, host, port, cmd, timeout=120):
        if "import run_farm" in cmd:
            return (0, "")
        if "worker-ready" in cmd:
            return (0, "/tmp/worker-ready")
        leg_calls["n"] += 1                       # the leg command
        return (1, "host dying") if leg_calls["n"] == 1 else (0, "done")

    monkeypatch.setattr(fleet, "_ssh", stateful_ssh)
    prov = FakeProvider([_offer("dead"), _offer("ok")], dead_ids={"dead"})
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "OK"                        # recovered on the fresh host
    assert prov.rented == ["dead", "ok"] and prov.live == {}   # failed over, both gone


def test_live_host_nonzero_is_terminal_run_fail(patched, tmp_path):
    """A non-zero exit on a host `dead_reason` says is ALIVE is a genuine work
    failure -- terminal RUN_FAIL, NOT a failover (don't re-run real failures)."""
    patched(ready=True, run_rc=1, run_out="boom")
    prov = FakeProvider([_offer("a"), _offer("b")])   # neither is dead
    [r] = _exec(prov, tmp_path).run([_leg("L1")])
    assert r.status == "RUN_FAIL" and prov.rented == ["a"]   # no failover to b


def test_fetch_strips_trailing_slash_to_match_marker(patched, tmp_path, monkeypatch):
    """A trailing-slash fetch must still land where marker() looks: the remote is
    rstrip'd so scp deposits leg_dir/<basename> == marker (else a successful leg
    wrongly reports NO_RESULT and re-runs)."""
    import os
    captured = {}

    def cap_scp_down(key, host, port, remote, dst):
        captured["remote"] = remote               # scp -r remote dst -> dst/<basename>
        os.makedirs(os.path.join(dst, os.path.basename(remote)), exist_ok=True)
        return (0, "")

    monkeypatch.setattr(fleet, "_scp_down", cap_scp_down)
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    leg = _leg("L1", fetch="out_kick/")            # trailing slash
    [r] = _exec(prov, tmp_path).run([leg])
    assert not captured["remote"].endswith("/")    # rstrip'd before scp
    assert os.path.basename(captured["remote"]) == leg.marker() == "out_kick"
    assert r.status == "OK"                         # marker found where scp landed


def test_post_run_sweep_destroys_stranded_rental(patched, tmp_path, monkeypatch):
    """If a rental escapes its leg's _untrack (a teardown gap), run()'s post-run
    reconciliation destroys it before it bills."""
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    ex = _exec(prov, tmp_path)
    monkeypatch.setattr(ex, "_untrack", lambda iid: None)   # simulate the tracking gap
    ex.run([_leg("L1")])
    assert "1000" in prov.destroyed                # swept by the post-run _destroy_live


# -- observability: progress streaming + parsing -----------------------------

@pytest.mark.parametrize("line,expected", [
    ("[[progress phase=kick frac=0.40 t=12s]]",
     {"phase": "kick", "frac": "0.40", "t": "12s"}),
    ("noise before [[progress phase=relax]] and after", {"phase": "relax"}),
    ("a plain log line, no marker", None),
    ("[[progress]]", {}),                                   # marker, no fields
    ("[[progress phase=kick frac=0.5", None),               # unterminated -> ignored
])
def test_parse_progress_line(line, expected):
    assert fleet.parse_progress_line(line) == expected


def _fake_ssh_stream_factory(lines, *, rc=0):
    """Emulate `_ssh_stream`: tee `lines` to progress_path, return (rc, joined)."""
    def fake(key, host, port, cmd, timeout, progress_path):
        with open(progress_path, "a") as f:
            for ln in lines:
                f.write(ln if ln.endswith("\n") else ln + "\n")
        return rc, "".join(ln + "\n" for ln in lines)
    return fake


def test_stream_progress_tees_log_and_progress_returns_last_marker(
        patched, monkeypatch, tmp_path):
    patched(ready=True)
    lines = ["booting", "[[progress phase=relax frac=0.0]]",
             "[[progress phase=kick frac=0.5 t=99s]]", "wrote manifest"]
    monkeypatch.setattr(fleet, "_ssh_stream", _fake_ssh_stream_factory(lines))
    leg = FleetLeg(label="L1", command="run.sh", ship=("driver.py",),
                   stream_progress=True)
    ex = _exec(FakeProvider([_offer("a")]), tmp_path)
    [r] = ex.run([leg])
    assert r.status == "OK"
    plog = tmp_path / "L1" / fleet.PROGRESS_LOG
    assert plog.exists() and "phase=kick" in plog.read_text()   # tee'd live
    # accessor returns the LAST structured marker, skipping the trailing free text
    assert ex.progress(leg) == {"phase": "kick", "frac": "0.5", "t": "99s"}


def test_non_streaming_leg_never_calls_ssh_stream(patched, monkeypatch, tmp_path):
    """Default legs are unchanged: plain `_ssh`, no progress.log, progress()=None."""
    patched(ready=True)
    monkeypatch.setattr(fleet, "_ssh_stream", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("_ssh_stream called for a non-streaming leg")))
    leg = _leg("L1")
    ex = _exec(FakeProvider([_offer("a")]), tmp_path)
    [r] = ex.run([leg])
    assert r.status == "OK"
    assert ex.progress(leg) is None
    assert not (tmp_path / "L1" / fleet.PROGRESS_LOG).exists()


# -- resume (Tier 1): restore-on-retry + incremental fetch -------------------

def test_restore_pushes_local_partial_only_when_present(monkeypatch, tmp_path):
    ups = []
    monkeypatch.setattr(fleet, "_scp_up",
                        lambda key, h, p, local, remote, **k: (ups.append((local, remote)), (0, ""))[1])
    ex = _exec(FakeProvider([]), tmp_path)
    leg = FleetLeg(label="L1", command="x", fetch="out_kick", resumable=True)
    host = types.SimpleNamespace(id="1", ssh_host="h", ssh_port=22)
    ex._restore(host, leg)                                  # nothing local yet
    assert ups == []
    part = tmp_path / "L1" / "out_kick"
    part.mkdir(parents=True)
    (part / "bare_relaxed.npz").write_text("x")
    ex._restore(host, leg)                                  # now restore fires
    assert len(ups) == 1
    local, remote = ups[0]
    assert local.endswith("L1/out_kick")                   # the accumulated partial
    assert remote == ex.remote_work_dir + "/"              # back to the same remote path


def test_fetch_loop_pulls_until_stopped(monkeypatch, tmp_path):
    ex = _exec(FakeProvider([]), tmp_path, fetch_interval_s=0.001)
    calls = []
    stop = threading.Event()

    def fake_fetch(host, leg):
        calls.append(1)
        if len(calls) >= 2:
            stop.set()
    monkeypatch.setattr(ex, "_fetch", fake_fetch)
    ex._fetch_loop(None, FleetLeg(label="L1", command="x", resumable=True), stop)
    assert len(calls) >= 2                                  # ticked, then stopped


def test_resumable_leg_restores_then_fetches_end_to_end(patched, monkeypatch, tmp_path):
    patched(ready=True)
    ups, downs = [], []
    monkeypatch.setattr(fleet, "_scp_up", lambda *a, **k: (ups.append(a), (0, ""))[1])
    monkeypatch.setattr(fleet, "_scp_down", lambda *a, **k: (downs.append(a), (0, ""))[1])
    part = tmp_path / "L1" / "out_kick"
    part.mkdir(parents=True)
    (part / "x.npz").write_text("x")                       # seed a partial -> restore fires
    leg = FleetLeg(label="L1", command="run.sh", ship=("driver.py",),
                   fetch="out_kick", done_when="out_kick/manifest.json",
                   resumable=True)
    ex = _exec(FakeProvider([_offer("a")]), tmp_path, fetch_interval_s=0.001)
    ex.run([leg])
    assert any(str(a[3]).endswith("L1/out_kick") for a in ups)   # restore push happened
    assert downs                                                  # fetched off-box


def test_non_resumable_leg_neither_restores_nor_background_fetches(
        patched, monkeypatch, tmp_path):
    """Default (atomic) leg: no restore, no mid-run fetch loop -- only the single
    final fetch. _restore must never be called."""
    patched(ready=True)
    monkeypatch.setattr(fleet, "_scp_down", lambda *a, **k: (0, ""))
    called = []
    monkeypatch.setattr(FleetExecutor, "_restore",
                        lambda self, h, leg: called.append(leg.label))
    leg = _leg("L1", fetch="out", done_when="out/manifest.json")
    _exec(FakeProvider([_offer("a")]), tmp_path).run([leg])
    assert called == []                                    # _restore never invoked
