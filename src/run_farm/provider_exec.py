"""ProviderExecutor: run a campaign over a rented fleet from any `Provider` (F).

The executor (D) that consumes the Provider seam: pull offers, rent a host with
per-host **failover** (a bad host -> next offer), wait for the engine to come up,
ship each config and run the campaign `worker` over SSH, sync the artifacts back,
and rely on the Provider's teardown-verifying `rent()` for teardown. This is the
principled generalization of the hand-rolled `run_eps_fleet` driver -- it works
over `VastProvider`, `RunPodProvider`, or any future `Provider`.

The physics crosses by name (`run_fn_ref`, a ``'module:function'`` ref the box
imports), never as a closure; configs travel as JSON on the command line. The
box must have run-farm + the engine providing `run_fn_ref` installed by the
`LaunchSpec.onstart` bootstrap; readiness probes `engine_module` (default: the
RunFn's own module).

v1 runs all configs sequentially on a single rented host. Multi-host parallel
fan-out (the `run_eps_fleet` ThreadPool pattern) is a documented follow-up.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from collections.abc import Iterable

from run_farm.protocols import (
    HostProbeFailed,
    HostSpec,
    LaunchSpec,
    Provider,
    RentedHost,
    RentUnavailable,
)
from run_farm.remote import ConfigClassRef, RunFnRef
from run_farm.worker import RESULT_PREFIX
from run_farm.protocols import RunConfig

DEFAULT_KEY = "~/.ssh/vastai"
# Matches the engine_dogfood vast/onstart.sh, which builds the engine into this env.
DEFAULT_REMOTE_PYTHON = "/workspace/jaxenv/bin/python"

# Shared ssh/scp `-o` options for every box connection (factored so the set can't
# drift across the four helpers -- which is how the missing keepalive arose, #43).
# ConnectTimeout only bounds CONNECTION setup; ServerAliveInterval adds a liveness
# check on a RUNNING session, so a host that dies or goes unreachable MID-command
# surfaces as a non-zero exit in ~ServerAliveInterval*ServerAliveCountMax (~2 min)
# instead of hanging until run_timeout (default 9000s) while the box may keep
# billing -- the leg then fails over / relaunches promptly.
_SSH_OPTS = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15",
             "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4"]


def _ssh(key: str, host: str, port: int, cmd: str, timeout: float = 120):  # pragma: no cover
    """Run one command on the box; returns (rc, combined stdout+stderr).

    Live subprocess to a rented host; tests monkeypatch this, so its body isn't
    unit-covered. A `TimeoutExpired` is returned AS a non-zero (rc, out) so the
    caller's normal rc!=0 handling applies (engine-ready retry, per-config error
    record) instead of the exception aborting the whole campaign."""
    try:
        r = subprocess.run(
            ["ssh", "-i", os.path.expanduser(key), *_SSH_OPTS,
             "-p", str(port), f"root@{host}", cmd],
            capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"ssh timeout after {timeout}s: {e}"


def _ssh_stream(key: str, host: str, port: int, cmd: str, timeout: float,
                progress_path: str):  # pragma: no cover  (live subprocess)
    """Like `_ssh`, but tee the box's stdout+stderr to `progress_path` line by
    line AS IT ARRIVES, so a multi-hour leg is observable live (`tail -f` the
    file on the control plane) instead of going dark until the command exits.

    Same return contract as `_ssh` -- `(rc, combined-output)`, with a timeout
    surfaced as `(124, ...)` so the caller's normal rc!=0 handling applies. A
    background reader thread pumps the pipe (so `proc.wait(timeout=...)` keeps
    clean timeout semantics regardless of whether the box is emitting output).
    Live subprocess to a rented host; tests monkeypatch this."""
    args = ["ssh", "-i", os.path.expanduser(key), *_SSH_OPTS,
            "-p", str(port), f"root@{host}", cmd]
    buf: list[str] = []
    try:
        proc = subprocess.Popen(args, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
    except OSError as e:
        return 1, f"ssh stream spawn error: {e}"

    def _pump(stream, sink):
        for line in stream:
            buf.append(line)
            sink.write(line)

    with open(progress_path, "a", buffering=1) as pf:
        reader = threading.Thread(target=_pump, args=(proc.stdout, pf), daemon=True)
        reader.start()
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            reader.join(timeout=5)
            return 124, "".join(buf) + f"\nssh stream timeout after {timeout}s"
        reader.join(timeout=10)
    return rc, "".join(buf)


def _scp_down(key: str, host: str, port: int, remote: str, local: str,
              timeout: float = 600):  # pragma: no cover  (live subprocess)
    try:
        r = subprocess.run(
            ["scp", "-i", os.path.expanduser(key), *_SSH_OPTS,
             "-P", str(port), "-r", f"root@{host}:{remote}", local],
            capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"scp timeout after {timeout}s: {e}"  # best-effort sync; don't abort


def _scp_up(key: str, host: str, port: int, local: str, remote: str,
            timeout: float = 600):  # pragma: no cover  (live subprocess)
    """Ship one local path up to the box (the FleetExecutor driver-script step)."""
    try:
        r = subprocess.run(
            ["scp", "-i", os.path.expanduser(key), *_SSH_OPTS,
             "-P", str(port), "-r", local, f"root@{host}:{remote}"],
            capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired as e:
        return 124, f"scp-up timeout after {timeout}s: {e}"


class ProviderExecutor:
    """Run a campaign over hosts rented from a `Provider`, with failover."""

    def __init__(self, provider: Provider, run_fn_ref: RunFnRef,
                 launch: LaunchSpec, *, host_spec: HostSpec | None = None,
                 key_path: str = DEFAULT_KEY,
                 remote_python: str = DEFAULT_REMOTE_PYTHON,
                 remote_work_dir: str = "/workspace/runs",
                 local_work_dir: str = "campaign_out",
                 ready_timeout: float = 900, run_timeout: float = 3600,
                 rent_timeout: float = 600, max_attempts: int = 12,
                 engine_module: str | None = None,
                 config_class: ConfigClassRef | None = None):
        self.provider = provider
        # Readiness probes the RunFn's OWN module by default: if `pkg.mod:fn`
        # imports, both the engine AND the exact entry point the worker will call
        # are present -- strictly stronger than probing a top-level package, and it
        # needs no configuration. Override for an engine whose RunFn lives somewhere
        # the bootstrap installs separately.
        self.engine_module = engine_module or run_fn_ref.split(":", 1)[0]
        # None -> derive per-config from the object the driver planned with.
        self.config_class = config_class
        # Distinct per backing provider so a multi-provider harvest can tell
        # Vast from RunPod (both are ProviderExecutors): "provider:vast" etc.
        self.name = f"provider:{getattr(provider, 'name', 'unknown')}"
        self.run_fn_ref = run_fn_ref
        self.launch = launch
        self.host_spec = host_spec or HostSpec()
        self.key_path = key_path
        self.remote_python = remote_python
        self.remote_work_dir = remote_work_dir
        self.local_work_dir = local_work_dir
        self.ready_timeout = ready_timeout
        self.run_timeout = run_timeout
        self.rent_timeout = rent_timeout
        # Cap the failover walk: trying 200 marginal offers at ready_timeout each
        # is a multi-day grind before giving up; bound it to the best N (#48).
        self.max_attempts = max_attempts

    # -- per-host steps ------------------------------------------------------
    def _wait_engine_ready(self, host: RentedHost) -> None:
        """Block until `import <engine_module>` succeeds on the box (the onstart
        bootstrap finished), or raise HostProbeFailed so the run fails over.

        The Provider's rent() only guarantees the host is SSH-reachable; the
        engine install (onstart) may still be running, so probe for it (P9).

        ⚠ COST NOTE: this probe cannot distinguish "not ready yet" from "never
        going to be ready", so a BROKEN bootstrap burns the full `ready_timeout`
        before failover -- measured 2026-07-16 at ~1067s vs 119s for the
        sentinel-with-a-bad-marker path (`fleet.SentinelReady`), i.e. ~9x the cost
        for the same verdict. Prefer an onstart that writes an explicit failure
        marker where you can. The asymmetry gets worse at grant scale, where
        `ready_timeout` is raised for slow engine installs."""
        check = f"{self.remote_python} -c 'import {self.engine_module}'"
        deadline = time.monotonic() + self.ready_timeout
        last = ""
        while time.monotonic() < deadline:
            rc, out = _ssh(self.key_path, host.ssh_host, host.ssh_port, check,
                           timeout=30)
            if rc == 0:
                return
            last = out
            time.sleep(10)
        raise HostProbeFailed(
            f"engine not ready on {host.id} within {self.ready_timeout}s: "
            f"{last[-160:]}")

    def _config_class_ref(self, config: RunConfig) -> str:
        """The concrete config type, by name, so the worker rebuilds the right one.

        Derived from the object by default: the driver planned with it, so it is
        always right and needs no configuration. Override via `config_class=` when
        the class isn't importable under its own `__module__` -- a config defined in
        `__main__`, or in a test module.
        """
        t = type(config)
        return self.config_class or f"{t.__module__}:{t.__qualname__}"

    def _run_config(self, host: RentedHost, config: RunConfig) -> dict:
        """Run the worker for one config over SSH; parse its result record."""
        cmd = (f"{self.remote_python} -m run_farm.worker "
               f"--config-json {shlex.quote(config.to_json())} "
               f"--run-fn {shlex.quote(self.run_fn_ref)} "
               f"--config-class {shlex.quote(self._config_class_ref(config))} "
               f"--work-dir {shlex.quote(self.remote_work_dir)}")
        rc, out = _ssh(self.key_path, host.ssh_host, host.ssh_port, cmd,
                       timeout=self.run_timeout)
        for line in out.splitlines():
            if line.startswith(RESULT_PREFIX):
                payload = line[len(RESULT_PREFIX):]
                # The worker runs on an untrusted remote box and its stdout can be
                # truncated or interleaved; a malformed result line is this one
                # config's failure, not grounds to abort the whole campaign.
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as e:
                    return {"run": config.run_name(), "result": None,
                            "skipped": False,
                            "error": f"malformed result line ({e}): {payload[:200]}"}
        return {"run": config.run_name(), "result": None, "skipped": False,
                "error": f"rc={rc}: {out[-200:]}"}

    def _sync_back(self, host: RentedHost) -> None:
        """Best-effort: pull the run artifacts (checkpoints/events/triggers)."""
        os.makedirs(self.local_work_dir, exist_ok=True)
        _scp_down(self.key_path, host.ssh_host, host.ssh_port,
                  self.remote_work_dir, self.local_work_dir)

    # -- the campaign --------------------------------------------------------
    def run(self, configs: Iterable[RunConfig], *, admission=None) -> list[dict]:
        """Rent a host (failing over bad ones), run every config on it, sync the
        artifacts back, and tear down. Returns the per-config result records.

        Failover covers provisioning AND mid-run: a config error on a host that
        `dead_reason` confirms is dead re-rents a fresh host and re-runs (the
        worker's resume skips checkpointed work), rather than absorbing the death
        as per-config errors. The failover walk is capped at `max_attempts` offers
        so a marketplace of marginal hosts can't grind for days (#48).

        Teardown is the Provider's teardown-verifying `rent()` invariant -- it fires on
        every exit, including the failover `continue` and any exception here.

        `admission` is enforced structurally here -- by the Provider's
        `offers()`/`rent()` host gates (the `HostSpec`), not a campaign
        `Admission`. A non-None campaign Admission would be silently ignored, so
        reject it rather than letting a caller believe it is applied.
        """
        if admission is not None:
            raise NotImplementedError(
                "ProviderExecutor enforces host admission via the Provider's "
                "offers()/rent() gates (HostSpec), not a campaign Admission; "
                "pass admission=None")
        configs = list(configs)
        if not configs:
            return []
        offers = self.provider.offers(self.host_spec)
        if not offers:
            raise RuntimeError(
                f"{self.provider.name}: no offers match {self.host_spec}")
        attempts = offers[:self.max_attempts]            # bound the failover walk (#48)
        last_err: Exception | None = None
        for offer in attempts:
            try:
                with self.provider.rent(offer, self.launch,
                                        timeout_s=self.rent_timeout) as host:
                    self._wait_engine_ready(host)        # may raise -> failover
                    results = []
                    for c in configs:
                        rec = self._run_config(host, c)
                        # If a config errors AND the provider confirms the host is
                        # dead, this is a HOST failure, not a config failure: fail
                        # over to a fresh host instead of hammering a corpse for a
                        # full run_timeout on every remaining config (#48). The
                        # worker's own resume skips work already checkpointed.
                        if rec.get("error") and self._host_dead(host.id):
                            raise HostProbeFailed(
                                f"host {host.id} died mid-run on "
                                f"{c.run_name()}: {str(rec['error'])[:120]}")
                        results.append(rec)
                    self._sync_back(host)
                    return results
            except (HostProbeFailed, TimeoutError, RentUnavailable) as e:
                last_err = e                              # bad host/race -> next offer
                continue
        raise RuntimeError(
            f"{self.provider.name}: all {len(attempts)} offer attempt(s) failed to "
            f"run (of {len(offers)} matched); last error: {last_err}")

    def _host_dead(self, host_id: str) -> str | None:
        """The provider's `dead_reason` for `host_id` if it exposes one and the
        host has visibly died, else None (provider-agnostic: no dead_reason or a
        transient read -> None = not-confirmed-dead, so a config error stays a
        config failure, not a spurious failover)."""
        dr = getattr(self.provider, "dead_reason", None)
        if dr is None:                                    # pragma: no cover
            return None
        try:
            return dr(host_id)
        except Exception:                                 # noqa: BLE001  # pragma: no cover
            return None
