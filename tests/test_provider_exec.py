"""ProviderExecutor tests: a FakeProvider (no spend) + monkeypatched SSH cover
the campaign-over-rented-fleet path -- happy run, per-host failover, and the
leak-proof teardown that must fire on every exit.
"""

import contextlib
import json

import pytest

import run_farm.provider_exec as pe
from run_farm import (HostProbeFailed, HostSpec, LaunchSpec, Offer,
                                    RentedHost)
from run_farm.provider_exec import ProviderExecutor
from run_farm.worker import RESULT_PREFIX
from run_farm import SimpleRunConfig as RunConfig

LAUNCH = LaunchSpec(image="img:12.2", onstart="echo hi", disk_gb=24)
SPEC = HostSpec(gpu_name="RTX 3090", max_dph=0.30)
CONFIGS = [RunConfig(name="faddeev_cp1", params={"R": 2.6}),
           RunConfig(name="faddeev_cp1", params={"R": 3.0})]


def _offer(oid, dph=0.12):
    return Offer(id=oid, dph=dph, gpu_name="RTX 3090", num_gpus=1,
                 reliability=0.99, inet_down_mbps=800, cuda_max=12.4,
                 geolocation="x", provider="fake")


class FakeProvider:
    """offers() + leak-proof rent() with a `live` set; bad_ids fail to come up."""

    name = "fake"

    def __init__(self, offers, bad_ids=(), dead_ids=()):
        self._offers = list(offers)
        self._bad = set(bad_ids)
        self._dead = set(dead_ids)            # hosts dead_reason will flag dead
        self.live: set[str] = set()
        self.rented: list[str] = []
        self._inst_offer: dict[str, str] = {}
        self._n = 0

    def offers(self, spec):
        return list(self._offers)

    @contextlib.contextmanager
    def rent(self, offer, launch, *, timeout_s=600):
        iid = f"inst-{self._n}"; self._n += 1
        self.rented.append(offer.id); self.live.add(iid)
        self._inst_offer[iid] = offer.id
        try:
            if offer.id in self._bad:
                raise HostProbeFailed(f"bad host {offer.id}")
            yield RentedHost(id=iid, ssh_host="10.0.0.1", ssh_port=22,
                             offer=offer)
        finally:
            self.live.discard(iid)            # leak-proof teardown

    def dead_reason(self, instance_id):
        oid = self._inst_offer.get(str(instance_id))
        return f"dead host {oid}" if oid in self._dead else None


def _fake_ssh_factory(*, ready=True, run_rc=0, worker_out=None):
    """Build a fake _ssh: the 'import pkg.mod' readiness probe and the
    worker invocation both routed by inspecting the command string."""
    def fake_ssh(key, host, port, cmd, timeout=120):
        if "import pkg.mod" in cmd:
            return (0, "") if ready else (1, "ModuleNotFoundError")
        if "run_farm.worker" in cmd:
            if worker_out is not None:
                return (run_rc, worker_out)
            rec = {"run": "r", "result": {"ok": True}, "skipped": False}
            return (run_rc, RESULT_PREFIX + json.dumps(rec) + "\n")
        return (0, "")
    return fake_ssh


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))

    def apply(**kw):
        monkeypatch.setattr(pe, "_ssh", _fake_ssh_factory(**kw))
    return apply


def test_runs_all_configs_and_tears_down(patched):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=1)
    results = ex.run(CONFIGS)
    assert len(results) == 2 and all(r["result"] == {"ok": True} for r in results)
    assert prov.live == set()                 # host torn down
    assert prov.rented == ["a"]


def test_fails_over_bad_host(patched):
    patched(ready=True)
    prov = FakeProvider([_offer("bad"), _offer("good")], bad_ids={"bad"})
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=1)
    results = ex.run(CONFIGS)
    assert len(results) == 2                   # ran on the good host
    assert prov.rented == ["bad", "good"]      # failed over past the bad one
    assert prov.live == set()                  # both attempts torn down


def test_engine_never_ready_fails_over_then_raises(patched):
    patched(ready=False)                       # import check always fails
    prov = FakeProvider([_offer("a"), _offer("b")])
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=0.05)
    with pytest.raises(RuntimeError, match="all .* offer attempt"):
        ex.run(CONFIGS)
    assert prov.rented == ["a", "b"]           # tried both, both timed out ready
    assert prov.live == set()                  # neither leaked


def test_malformed_result_line_is_error_record_not_crash(patched, monkeypatch):
    """A truncated/garbled worker result line is one config's error, not a
    campaign-aborting JSONDecodeError."""
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))
    monkeypatch.setattr(pe, "_ssh", _fake_ssh_factory(
        ready=True, worker_out=RESULT_PREFIX + '{"run": "r", "result": '))  # truncated
    prov = FakeProvider([_offer("a")])
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=1)
    results = ex.run(CONFIGS)
    assert len(results) == 2
    assert all("malformed result line" in r["error"] for r in results)
    assert prov.live == set()                   # still torn down


def test_non_none_admission_rejected(patched):
    """ProviderExecutor enforces admission via the Provider's host gates; a
    campaign Admission would be silently ignored, so reject it loudly."""
    patched(ready=True)
    ex = ProviderExecutor(FakeProvider([_offer("a")]), "pkg.mod:fn", LAUNCH,
                          host_spec=SPEC)
    with pytest.raises(NotImplementedError, match="admission"):
        ex.run(CONFIGS, admission=object())


def test_no_offers_raises(patched):
    patched(ready=True)
    ex = ProviderExecutor(FakeProvider([]), "pkg.mod:fn", LAUNCH, host_spec=SPEC)
    with pytest.raises(RuntimeError, match="no offers"):
        ex.run(CONFIGS)


def test_empty_configs_is_noop(patched):
    patched(ready=True)
    prov = FakeProvider([_offer("a")])
    assert ProviderExecutor(prov, "pkg.mod:fn", LAUNCH).run([]) == []
    assert prov.rented == []                    # never even rented


def test_max_attempts_caps_the_failover_walk(patched):
    """A marketplace of marginal hosts must not grind for days: the failover walk
    stops after max_attempts offers, not all of them (CM #48)."""
    patched(ready=False)                        # every host fails the ready probe
    prov = FakeProvider([_offer(f"o{i}") for i in range(10)])
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=0.02, max_attempts=3)
    with pytest.raises(RuntimeError, match="offer attempt"):
        ex.run(CONFIGS)
    assert prov.rented == ["o0", "o1", "o2"]    # capped at 3, not all 10


def test_mid_run_failover_on_confirmed_host_death(patched, monkeypatch):
    """A config error on a host that `dead_reason` confirms is dead fails over to a
    fresh host and re-runs, instead of absorbing the death as per-config errors and
    hammering the corpse for the remaining configs (CM #48)."""
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))
    calls = {"n": 0}

    def stateful_ssh(key, host, port, cmd, timeout=120):
        if "import pkg.mod" in cmd:
            return (0, "")
        if "run_farm.worker" in cmd:
            calls["n"] += 1
            if calls["n"] == 1:                 # first host errors as it dies
                return (1, "boom -- no result line")
            rec = {"run": "r", "result": {"ok": True}, "skipped": False}
            return (0, RESULT_PREFIX + json.dumps(rec) + "\n")
        return (0, "")

    monkeypatch.setattr(pe, "_ssh", stateful_ssh)
    prov = FakeProvider([_offer("dead"), _offer("good")], dead_ids={"dead"})
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=0.2)
    results = ex.run(CONFIGS)
    assert prov.rented == ["dead", "good"]      # failed over past the dead host
    assert all(r.get("result") for r in results)  # re-ran clean on the good host
    assert prov.live == set()                   # both torn down


def test_box_connections_carry_serveralive_keepalive(monkeypatch, tmp_path):
    """Every ssh/scp helper must carry ServerAliveInterval, so a host that dies
    or goes unreachable MID-command surfaces as a non-zero exit in ~2 min instead
    of hanging on run_timeout (#43). The live subprocess is otherwise no-cover, so
    capture the argv each helper builds and assert the liveness options are wired
    into all four box-connection paths."""
    seen = []

    class _FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

        def wait(self, timeout=None):
            return 0

    def fake_run(argv, **kw):
        seen.append(argv)
        return _FakeProc()

    def fake_popen(argv, **kw):
        seen.append(argv)
        p = _FakeProc()
        p.stdout = iter(())                      # no streamed lines
        return p

    monkeypatch.setattr(pe.subprocess, "run", fake_run)
    monkeypatch.setattr(pe.subprocess, "Popen", fake_popen)

    pe._ssh("k", "h", 22, "echo hi")
    pe._scp_down("k", "h", 22, "remote", str(tmp_path))
    pe._scp_up("k", "h", 22, str(tmp_path), "remote")
    pe._ssh_stream("k", "h", 22, "echo hi", 60, str(tmp_path / "p.log"))

    assert len(seen) == 4                        # all four helpers exercised
    for argv in seen:
        joined = " ".join(argv)
        assert "ServerAliveInterval=30" in joined, f"no keepalive: {argv}"
        assert "ServerAliveCountMax=4" in joined, f"no count-max: {argv}"
        assert "ConnectTimeout=15" in joined      # the setup bound is still there
