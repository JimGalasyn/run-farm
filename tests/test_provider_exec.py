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


# --------------------------------------------------------------- remote env ----
# `remote_env` exists because a non-interactive `ssh host cmd` sources no profile:
# whatever the provider's onstart exported is NOT in the worker's environment. The
# motivating case is XLA_FLAGS, which JAX reads when the backend initialises, so
# setting it from inside the RunFn is already too late.

def _capturing_ssh(seen):
    """Record every command string, answering probe and worker normally."""
    def fake_ssh(key, host, port, cmd, timeout=120):
        seen.append(cmd)
        if "import pkg.mod" in cmd:
            return (0, "")
        if "run_farm.worker" in cmd:
            rec = {"run": "r", "result": {"ok": True}, "skipped": False}
            return (0, RESULT_PREFIX + json.dumps(rec) + "\n")
        return (0, "")
    return fake_ssh


def test_remote_env_reaches_the_worker_command(monkeypatch):
    seen = []
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))
    monkeypatch.setattr(pe, "_ssh", _capturing_ssh(seen))

    ex = ProviderExecutor(FakeProvider([_offer("a")]), "pkg.mod:fn", LAUNCH,
                          host_spec=SPEC, ready_timeout=1,
                          remote_env={"XLA_FLAGS": "--xla_gpu_autotune_level=0"})
    ex.run(CONFIGS)

    worker_cmds = [c for c in seen if "run_farm.worker" in c]
    assert worker_cmds, "the worker never ran"
    for cmd in worker_cmds:
        assert cmd.startswith("env "), f"env prefix missing: {cmd[:80]}"
        # No shell metacharacters in this value, so shlex.quote leaves it bare.
        assert "XLA_FLAGS=--xla_gpu_autotune_level=0 " in cmd


def test_remote_env_also_applies_to_the_readiness_probe(monkeypatch):
    """A var that breaks `import` must fail readiness, not every leg after it.

    If the probe ran bare, a bad XLA_FLAGS would pass readiness and then fail the
    engine on each config -- discovered at rental prices instead of at the probe.
    """
    seen = []
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))
    monkeypatch.setattr(pe, "_ssh", _capturing_ssh(seen))

    ex = ProviderExecutor(FakeProvider([_offer("a")]), "pkg.mod:fn", LAUNCH,
                          host_spec=SPEC, ready_timeout=1,
                          remote_env={"XLA_FLAGS": "--xla_gpu_autotune_level=0"})
    ex.run(CONFIGS)

    probes = [c for c in seen if "import pkg.mod" in c]
    assert probes, "readiness never probed"
    assert all(c.startswith("env XLA_FLAGS=") for c in probes)


def test_remote_env_values_are_shell_quoted():
    """XLA_FLAGS is a space-separated list; unquoted, its tail becomes a command."""
    ex = ProviderExecutor(FakeProvider([_offer("a")]), "pkg.mod:fn", LAUNCH,
                          host_spec=SPEC,
                          remote_env={"XLA_FLAGS": "--a=0 --b=1", "SAFE": "x"})
    out = ex._envify("python -m thing")
    # Both flags stay inside ONE argument.
    assert "XLA_FLAGS='--a=0 --b=1'" in out
    assert out.endswith("python -m thing")
    # Sorted by key, so the command string is stable and diffable across runs.
    assert out.index("SAFE=") < out.index("XLA_FLAGS=")


def test_no_remote_env_leaves_the_command_untouched():
    """The default path must not grow an `env` prefix -- this is the regression
    guard for every existing consumer."""
    ex = ProviderExecutor(FakeProvider([_offer("a")]), "pkg.mod:fn", LAUNCH,
                          host_spec=SPEC)
    assert ex._envify("python -m thing") == "python -m thing"


# ------------------------------------------------- transport reattach (#2) ----
# `run` already fails a config over when the provider CONFIRMS the host died. The
# case left uncovered was the opposite one: rc=255 over a host that is still ALIVE,
# recorded as a config error and lost with its checkpoint on a healthy box.

def _flaky_ssh(script, seen=None):
    """_ssh whose worker invocations return `script` in order; probes answer GONE."""
    calls = {"worker": 0, "probe": 0}

    def fake_ssh(key, host, port, cmd, timeout=120):
        if seen is not None:
            seen.append(cmd)
        if "import pkg.mod" in cmd:
            return (0, "")
        if "pgrep" in cmd:
            calls["probe"] += 1
            return (0, "GONE\n")
        if "run_farm.worker" in cmd:
            i = calls["worker"]
            calls["worker"] += 1
            return script[min(i, len(script) - 1)]
        return (0, "")
    fake_ssh.calls = calls
    return fake_ssh


_OK = (0, RESULT_PREFIX + json.dumps(
    {"run": "r", "result": {"ok": True}, "skipped": False}) + "\n")
_TRANSPORT = (255, "ssh: connection closed by remote host")


def _exec(monkeypatch, fake, **kw):
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))
    monkeypatch.setattr(pe, "_ssh", fake)
    return ProviderExecutor(FakeProvider([_offer("a")]), "pkg.mod:fn", LAUNCH,
                            host_spec=SPEC, ready_timeout=1, **kw)


def test_transport_failure_is_retried_when_the_host_is_alive(monkeypatch):
    """The bug: a dropped channel over a healthy box lost the leg."""
    fake = _flaky_ssh([_TRANSPORT, _OK])
    ex = _exec(monkeypatch, fake, reattach_attempts=2)
    results = ex.run(CONFIGS[:1])
    assert results[0]["result"] == {"ok": True}     # recovered, not lost
    assert fake.calls["worker"] == 2                # reattached exactly once


def test_reattach_is_off_by_default(monkeypatch):
    """Existing consumers keep the old single-attempt behaviour byte-for-byte."""
    fake = _flaky_ssh([_TRANSPORT, _OK])
    ex = _exec(monkeypatch, fake)                   # no reattach_attempts
    results = ex.run(CONFIGS[:1])
    assert fake.calls["worker"] == 1                # did NOT retry
    assert "rc=255" in results[0]["error"]


def test_a_real_worker_failure_is_not_retried(monkeypatch):
    """rc!=255 is the worker speaking. Trust it — retrying would hide a real bug."""
    fail = (7, "Traceback: your RunFn raised")
    fake = _flaky_ssh([fail, _OK])
    ex = _exec(monkeypatch, fake, reattach_attempts=3)
    results = ex.run(CONFIGS[:1])
    assert fake.calls["worker"] == 1
    assert "rc=7" in results[0]["error"]


def test_no_second_worker_is_started_while_one_may_still_run(monkeypatch):
    """The launch-if-not-already-running contract the worker CLI does not own.

    Two workers interleaving checkpoints into one registry dir is unrecoverable;
    a lost leg is not. So an ALIVE probe must decline to re-invoke.
    """
    def fake_ssh(key, host, port, cmd, timeout=120):
        if "import pkg.mod" in cmd:
            return (0, "")
        if "pgrep" in cmd:
            return (0, "ALIVE\n")               # a survivor is still running
        if "run_farm.worker" in cmd:
            fake_ssh.n += 1
            return _TRANSPORT
        return (0, "")
    fake_ssh.n = 0

    # Tiny backoff: `sleep` is a no-op under the fixture, so the wait budget is
    # spent in real wall time. 0.01 keeps the whole wait under ~30ms.
    ex = _exec(monkeypatch, fake_ssh, reattach_attempts=3,
               reattach_backoff_s=0.01)
    results = ex.run(CONFIGS[:1])
    assert fake_ssh.n == 1, "started a second worker over a live one"
    assert "declined to start a second one" in results[0]["error"]


def test_confirmed_dead_host_stops_reattaching(monkeypatch):
    """Don't reattach to a corpse — hand it back so `run` fails over."""
    prov = FakeProvider([_offer("a"), _offer("b")], dead_ids={"a"})
    fake = _flaky_ssh([_TRANSPORT, _OK])
    monkeypatch.setattr(pe.time, "sleep", lambda s: None)
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))
    monkeypatch.setattr(pe, "_ssh", fake)
    ex = ProviderExecutor(prov, "pkg.mod:fn", LAUNCH, host_spec=SPEC,
                          ready_timeout=1, reattach_attempts=3)
    ex.run(CONFIGS[:1])
    # One attempt on the dead host, then failover to offer b (which succeeds).
    assert prov.rented == ["a", "b"]


def test_reattach_shares_one_deadline_and_does_not_multiply_the_bill(monkeypatch):
    """Per-attempt run_timeouts would multiply the billing window by the attempts.

    `sleep` is deliberately NOT patched here, and the backoff is real-but-tiny, so
    wall time genuinely advances between attempts. Without that the budget shrinks
    only by nanoseconds and a per-attempt-timeout mutant passes — which it did on
    the first version of this test.
    """
    fake = _flaky_ssh([_TRANSPORT, _TRANSPORT, _TRANSPORT])
    seen = []
    monkeypatch.setattr(pe, "_scp_down", lambda *a, **k: (0, ""))

    def timing_ssh(key, host, port, cmd, timeout=120):
        if "run_farm.worker" in cmd:
            seen.append(timeout)
        return fake(key, host, port, cmd, timeout)

    monkeypatch.setattr(pe, "_ssh", timing_ssh)
    backoff = 0.05
    ex = ProviderExecutor(FakeProvider([_offer("a")]), "pkg.mod:fn", LAUNCH,
                          host_spec=SPEC, ready_timeout=1, run_timeout=100,
                          reattach_attempts=3, reattach_backoff_s=backoff)
    ex.run(CONFIGS[:1])

    assert len(seen) == 4, f"expected 1 + 3 reattaches, got {len(seen)}"
    assert seen[0] <= 100
    # STRICTLY decreasing, by at least one backoff each time: that is what
    # distinguishes one shared deadline from a fresh run_timeout per attempt.
    for a, b in zip(seen, seen[1:]):
        assert b < a - backoff / 2, f"budget did not shrink across attempts: {seen}"
