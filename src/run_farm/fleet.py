"""FleetExecutor: a parallel, one-host-per-leg *script* fleet over any Provider.

The 2026-06-15 farming session ran three near-identical hand-rolled drivers
(`run_eps_fleet`, `run_stability_fleet`, `run_eps_kick_fleet`), each copy-pasting
the same loop: pull an offer, rent it, wait for the worker, scp a driver script
up, ssh-run it, scp the output dir back, fail over bad hosts. `ProviderExecutor`
runs the *structured* campaign worker (one RunConfig -> one RunFn record) and
sequentially on a single host; it does not cover the "ship an arbitrary script +
args, fetch an output glob, one rented host per leg, in parallel" shape those
drivers need. `FleetExecutor` is that shape, factored once (#25): a fleet run is
**data** -- a list of `FleetLeg(label, command, ship, fetch)` -- not a forked
script, so the three drivers collapse to thin callers that build legs.

It is physics-agnostic (no model/stepper import): the only thing crossing the
boundary is a shell `command` the box runs and the files it ships/fetches. The
robustness the farming session paid for is built in:

  - per-leg failover on a bad/unreachable host (`HostProbeFailed`) or an offer
    race (`RentUnavailable`) -> pull the next offer;
  - **fast-fail a corpse** via the provider's API status (#27) instead of
    ssh-polling a container that never came up for the full deadline;
  - **in-flight host-death retry** (#48): if a leg's command exits non-zero AND
    the provider's `dead_reason` CONFIRMS the host died under it, the leg fails
    over and re-runs on a fresh host (legs are idempotent). A non-zero exit on a
    *live* host is a genuine work failure -- a terminal `RUN_FAIL`, not retried;
  - **refresh the offer pool** when it drains under heavy failover (#28);
  - **resume**: a leg whose output already exists locally is skipped (#26), so a
    whole-leg failure also recovers on relaunch;
  - **launch jitter** so N legs don't fire N simultaneous DNS lookups (#29);
  - **signal-safe teardown** (#24): a SIGTERM/SIGINT mid-run still destroys every
    in-flight rental, a backstop to each `rent()`'s own best-effort teardown;
  - **post-run reconciliation** (#48): `run()` sweeps any rental still tracked at
    the end (a teardown gap) before returning.

Recovery boundary, stated honestly: hosts are recovered at provisioning, between
legs, and mid-leg WHEN `dead_reason` can confirm the death; a host that dies in a
way `dead_reason` cannot detect surfaces as a `RUN_FAIL` recoverable by
relaunch+resume, not by silent in-flight re-renting.

Transient REST retry (#23) and the best-effort teardown live in the Provider
(`vast.py`), so they apply here for free.
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import os
import shlex
import signal
import threading
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path

from run_farm.protocols import (
    HostProbeFailed,
    HostSpec,
    LaunchSpec,
    LeakRisk,
    Provider,
    RentedHost,
    RentUnavailable,
)
from run_farm.provider_exec import (
    DEFAULT_KEY,
    _scp_down,
    _scp_up,
    _ssh,
    _ssh_stream,
)


@dataclasses.dataclass(frozen=True)
class FleetLeg:
    """One unit of fleet work: ship inputs, run a command, fetch outputs.

      label    unique id; names the local output subdir AND the resume key
      command  shell command run on the box, cwd = `remote_work_dir`
      ship     local paths scp'd up to `remote_work_dir` before the command
      fetch    remote path (relative to `remote_work_dir`, or absolute) scp'd
               back into `<local_out_dir>/<label>/` after a rc==0 command
      done_when  local path under `<local_out_dir>/<label>/` whose existence
               means "already complete" -- the resume/skip marker (#26). Defaults
               to the basename of `fetch`; a more precise marker (e.g.
               ``"out_kick/manifest.json"``) makes resume robust to a partial
               fetch.
      stream_progress  tee the box's stdout to `<local_out_dir>/<label>/
               progress.log` line by line as it arrives, so a multi-hour leg is
               observable live (`tail -f`) instead of going dark until exit. Off
               by default (atomic legs are byte-for-byte unchanged); opt in for
               long single-command legs. See `FleetExecutor.progress`.
      resumable  treat the leg as resumable, not atomic: pull `fetch` off-box on
               a cadence MID-RUN (not only at done), so partial results survive a
               host that dies after an hour; and on a retry, push the accumulated
               local partial back UP to the replacement box first, so the box's
               own skip-if-exists continues instead of recomputing. Off by
               default (atomic legs unchanged). Requires a driver that does
               skip-if-exists + atomic writes for the partial to be trustworthy.
    """

    label: str
    command: str
    ship: tuple[str, ...] = ()
    fetch: str = ""
    done_when: str = ""
    stream_progress: bool = False
    resumable: bool = False

    def marker(self) -> str:
        """The local relative path that signals this leg is complete."""
        return self.done_when or (os.path.basename(self.fetch.rstrip("/"))
                                  if self.fetch else "")


#: Filename of the live stdout tee under a leg's local dir (when stream_progress).
PROGRESS_LOG = "progress.log"


def parse_progress_line(line: str) -> dict[str, str] | None:
    """Parse one structured progress marker, or None if the line isn't one.

    The convention a driver emits to advertise where it is, e.g.::

        [[progress phase=kick frac=0.40 t=1234s]]

    Free-form text (the driver's normal logging) is ignored -- only lines
    containing a ``[[progress ...]]`` token are parsed, into a flat str->str
    dict of the ``key=value`` pairs. Physics-agnostic: keys are whatever the
    driver chose (`phase`, `frac`, `seed`, ...)."""
    i = line.find("[[progress")
    if i < 0:
        return None
    j = line.find("]]", i)
    if j < 0:
        return None
    body = line[i + len("[[progress"):j]
    out: dict[str, str] = {}
    for tok in body.split():
        k, _, v = tok.partition("=")
        if k:
            out[k] = v
    return out


@dataclasses.dataclass(frozen=True)
class LegResult:
    """The outcome of one leg. `status` is one of:
    OK | SKIP | NO_RESULT | RUN_FAIL | NO_OFFERS | LEAK | ERROR."""

    label: str
    status: str
    host_id: str | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("OK", "SKIP")


# --------------------------------------------------------------- readiness ----
# How a leg knows its box is ready to run work. The Provider's rent() only
# guarantees SSH-reachable; the onstart bootstrap (engine install, net probe)
# may still be running, so probe for it (P9). A probe returns True (ready) /
# False (not yet) or raises HostProbeFailed (the host announced it is bad).
class ImportReady:
    """Ready when `python -c 'import <module>'` succeeds on the box -- the engine
    bootstrap (onstart) has finished installing the engine."""

    def __init__(self, python: str = "/workspace/jaxenv/bin/python",
                 module: str = "run_farm"):
        self.python, self.module = python, module

    def check(self, ssh, host: RentedHost) -> bool:
        rc, _out = ssh(f"{self.python} -c 'import {self.module}'", timeout=30)
        return rc == 0


class SentinelReady:
    """Ready when an onstart-written sentinel file appears; a `bad` sentinel
    (the onstart net-probe bailing on a throttled host) raises HostProbeFailed so
    the leg fails over at once instead of waiting out the deadline."""

    def __init__(self, ok: str = "/tmp/worker-ready",
                 bad: str = "/tmp/worker-bad-network"):
        self.ok, self.bad = ok, bad

    def check(self, ssh, host: RentedHost) -> bool:
        probe = f"ls {shlex.quote(self.ok)} {shlex.quote(self.bad)} 2>/dev/null"
        _rc, out = ssh(probe, timeout=30)
        if self.bad and os.path.basename(self.bad) in out:
            raise HostProbeFailed("onstart net-probe bailed (throttled host)")
        return os.path.basename(self.ok) in out


# --------------------------------------------------------------- offer pool ----
class _OfferPool:
    """A thread-safe, cheapest-first offer queue that re-queries the Provider
    when it drains under failover (#28). `offers()` returns offers sorted asc, so
    popleft is the cheapest remaining. Refills are capped so a genuinely empty
    marketplace ends the run instead of spinning."""

    def __init__(self, provider: Provider, spec: HostSpec, *, max_refills: int = 3):
        self._provider, self._spec, self._max_refills = provider, spec, max_refills
        self._lock = threading.Lock()
        self._q: deque = deque(provider.offers(spec))
        self._refills = 0
        self.initial = len(self._q)

    def get(self):
        """Next offer, refilling once if drained; None when truly exhausted."""
        with self._lock:
            if not self._q and self._refills < self._max_refills:
                self._refills += 1
                self._q.extend(self._provider.offers(self._spec))
            return self._q.popleft() if self._q else None


# --------------------------------------------------------------- executor ----
class FleetExecutor:
    """Run a list of `FleetLeg`s, one rented host per leg, in parallel."""

    def __init__(self, provider: Provider, launch: LaunchSpec, *,
                 local_out_dir: str, host_spec: HostSpec | None = None,
                 ready=None, key_path: str = DEFAULT_KEY,
                 remote_work_dir: str = "/workspace",
                 max_parallel: int = 12, rent_timeout: float = 600,
                 ready_timeout: float = 1200, ready_poll_s: float = 15,
                 run_timeout: float = 9000, jitter_s: float = 2.0,
                 fetch_interval_s: float = 120,
                 max_refills: int = 3, ledger=None, log=print):
        self.provider = provider
        self.launch = launch
        self.local_out_dir = Path(local_out_dir)
        self.host_spec = host_spec or HostSpec()
        self.ready = ready or ImportReady()
        self.key_path = key_path
        self.remote_work_dir = remote_work_dir
        self.max_parallel = max(1, max_parallel)
        self.rent_timeout = rent_timeout
        self.ready_timeout = ready_timeout
        self.ready_poll_s = ready_poll_s
        self.run_timeout = run_timeout
        self.jitter_s = jitter_s
        # Cadence (s) at which a `resumable` leg's fetch dir is pulled off-box
        # MID-RUN, so partial results (relaxed checkpoint, completed sub-legs)
        # survive a host that dies after an hour. Restored to the replacement box.
        self.fetch_interval_s = fetch_interval_s
        self.max_refills = max_refills
        self.ledger = ledger
        self._log = log
        # Live rentals, for the signal-safe teardown backstop (#24): instance id
        # -> True while a leg holds the host. Guarded; touched from worker threads.
        self._live: dict[str, bool] = {}
        self._live_lock = threading.Lock()

    # -- resume (#26) --------------------------------------------------------
    def _leg_dir(self, leg: FleetLeg) -> Path:
        return self.local_out_dir / leg.label

    # -- observability -------------------------------------------------------
    def progress(self, leg: FleetLeg) -> dict[str, str] | None:
        """Latest parsed `[[progress ...]]` marker for a streaming leg, or None.

        Reads the live tee at `<leg_dir>/progress.log` and returns the last
        structured marker the driver emitted (see `parse_progress_line`); None
        if the leg isn't streaming, hasn't started, or has emitted no marker
        yet. A cheap point-in-time poll for a watcher/heartbeat -- it does not
        block or tail."""
        path = self._leg_dir(leg) / PROGRESS_LOG
        try:
            lines = path.read_text().splitlines()
        except OSError:
            return None
        for line in reversed(lines):
            parsed = parse_progress_line(line)
            if parsed is not None:
                return parsed
        return None

    def _complete(self, leg: FleetLeg) -> bool:
        """True if this leg's output already exists locally -- skip it on a
        relaunch. With no marker we cannot tell, so it is never pre-skipped."""
        marker = leg.marker()
        return bool(marker) and (self._leg_dir(leg) / marker).exists()

    # -- per-host steps ------------------------------------------------------
    def _wait_ready(self, host: RentedHost) -> None:
        """Block until the box is ready (the `ready` probe passes), failing over
        if it announces bad OR the provider's API reports it dead (#27)."""
        def ssh(cmd, timeout=30):
            return _ssh(self.key_path, host.ssh_host, host.ssh_port, cmd, timeout)

        dead_reason = getattr(self.provider, "dead_reason", None)
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.ready.check(ssh, host):              # may raise HostProbeFailed
                return
            if dead_reason is not None:                  # fast-fail a corpse (#27)
                reason = dead_reason(host.id)
                if reason:
                    raise HostProbeFailed(f"fast-fail {host.id}: {reason}")
            time.sleep(self.ready_poll_s)
        raise HostProbeFailed(
            f"worker not ready on {host.id} within {self.ready_timeout}s")

    def _ship(self, host: RentedHost, leg: FleetLeg) -> None:
        for src in leg.ship:
            rc, out = _scp_up(self.key_path, host.ssh_host, host.ssh_port,
                              str(src), self.remote_work_dir + "/")
            if rc != 0:
                raise HostProbeFailed(f"scp-up {src} -> {host.id} failed: {out[-160:]}")

    def _fetch(self, host: RentedHost, leg: FleetLeg) -> None:
        if not leg.fetch:
            return
        remote = (leg.fetch if leg.fetch.startswith("/")
                  else f"{self.remote_work_dir}/{leg.fetch}")
        # Strip any trailing slash: `scp -r src/ dst` copies the CONTENTS into dst,
        # but `scp -r src dst` lands it as `dst/<basename>` -- which is exactly
        # where marker() (also rstrip'd) looks. Without this, a trailing-slash fetch
        # deposits the files one level up from the marker and a *successful* leg
        # reports NO_RESULT, needlessly re-running on relaunch (CM review #48).
        remote = remote.rstrip("/")
        leg_dir = self._leg_dir(leg)
        leg_dir.mkdir(parents=True, exist_ok=True)
        _scp_down(self.key_path, host.ssh_host, host.ssh_port, remote, str(leg_dir))

    # -- resume transport (resumable legs) -----------------------------------
    def _restore(self, host: RentedHost, leg: FleetLeg) -> None:
        """Push any locally-accumulated partial `fetch` dir back UP to the box,
        so the driver's own skip-if-exists resumes from it instead of recomputing
        (resume on a *replacement* host). No-op if nothing was fetched yet."""
        if not leg.fetch:
            return
        local = self._leg_dir(leg) / os.path.basename(leg.fetch.rstrip("/"))
        if not local.exists():
            return
        # `fetch` is relative to remote_work_dir (or absolute). Land the dir back
        # at the SAME remote path: scp -r <local> <remote_parent>/ -> parent/<base>.
        remote = (leg.fetch if leg.fetch.startswith("/")
                  else f"{self.remote_work_dir}/{leg.fetch}").rstrip("/")
        remote_parent = os.path.dirname(remote)
        _scp_up(self.key_path, host.ssh_host, host.ssh_port,
                str(local), remote_parent + "/")

    def _fetch_loop(self, host: RentedHost, leg: FleetLeg,
                    stop: threading.Event) -> None:
        """Pull `fetch` off-box every `fetch_interval_s` until `stop`, so partial
        results accumulate locally even if the host dies mid-run. Best-effort: a
        failed pull (e.g. the dir doesn't exist yet) is swallowed -- the final
        `_fetch` and the leak-reaper are the backstops. The box writes atomically,
        so an in-flight pull never grabs a half-written file."""
        while not stop.wait(self.fetch_interval_s):
            try:
                self._fetch(host, leg)
            except Exception:                             # noqa: BLE001 best-effort
                pass

    # -- one leg, with failover ---------------------------------------------
    def _run_leg(self, pool: _OfferPool, leg: FleetLeg, idx: int) -> LegResult:
        # Launch jitter (#29): stagger starts within a parallel wave so we don't
        # fire max_parallel simultaneous resolver lookups (the thundering-herd
        # insurance, independent of any one cause). Deterministic (no RNG): the
        # i-th of each wave waits i*jitter.
        if self.jitter_s:
            time.sleep((idx % self.max_parallel) * self.jitter_s)
        tried = 0
        while True:
            offer = pool.get()
            if offer is None:
                return LegResult(leg.label, "NO_OFFERS", None,
                                 f"offer pool drained after {tried} attempt(s)")
            tried += 1
            try:
                with self.provider.rent(offer, self.launch,
                                        timeout_s=self.rent_timeout) as host:
                    self._track(host.id)
                    try:
                        self._wait_ready(host)
                        self._ship(host, leg)
                        if leg.resumable:
                            self._restore(host, leg)   # push prior partial up
                        cmd = f"cd {shlex.quote(self.remote_work_dir)} && {leg.command}"
                        fetch_stop = threading.Event()
                        fetcher = None
                        if leg.resumable:
                            fetcher = threading.Thread(
                                target=self._fetch_loop,
                                args=(host, leg, fetch_stop), daemon=True)
                            fetcher.start()
                        try:
                            if leg.stream_progress:
                                leg_dir = self._leg_dir(leg)
                                leg_dir.mkdir(parents=True, exist_ok=True)
                                rc, out = _ssh_stream(
                                    self.key_path, host.ssh_host, host.ssh_port, cmd,
                                    self.run_timeout, str(leg_dir / PROGRESS_LOG))
                            else:
                                rc, out = _ssh(self.key_path, host.ssh_host,
                                               host.ssh_port, cmd,
                                               timeout=self.run_timeout)
                        finally:
                            if fetcher is not None:
                                fetch_stop.set()
                                fetcher.join(timeout=30)
                        # host.id is the rented INSTANCE id (not offer.id), so a
                        # result correlates to the provider's live-instance list,
                        # `_track`, and `dead_reason`.
                        if rc != 0:
                            # Distinguish "the work failed" from "the host died
                            # under the work" (#48): if the provider confirms the
                            # host is dead, this is a host failure -- fail over and
                            # retry the leg on a fresh host (our legs are idempotent:
                            # per-seed/model outputs, resume skips completed).
                            # Otherwise the work genuinely failed -- a terminal
                            # RUN_FAIL, don't re-run it on new hardware.
                            dead = self._host_dead(host.id)
                            if dead:
                                raise HostProbeFailed(
                                    f"host {host.id} died mid-run: {dead}")
                            return LegResult(leg.label, "RUN_FAIL", host.id,
                                             f"rc={rc}: {out[-240:]}")
                        self._fetch(host, leg)
                        done = self._complete(leg) or not leg.marker()
                        return LegResult(leg.label, "OK" if done else "NO_RESULT",
                                         host.id)
                    finally:
                        self._untrack(host.id)
            except (HostProbeFailed, TimeoutError, RentUnavailable) as e:
                self._log(f"  {leg.label}: offer {offer.id} "
                          f"{type(e).__name__} -> failing over")
                continue                                  # bad host / race -> next
            except Exception as e:                        # noqa: BLE001
                # No host handle here (rent may have failed before yielding), so
                # the offer id is the best correlation we have for a terminal error.
                # A leak is flagged by TYPE (LeakRisk), not by whether the word
                # "LEAK" survived into the message -- the substring is a defensive
                # fallback only, so a reworded message can't hide a billing GPU.
                msg = str(e)
                is_leak = isinstance(e, LeakRisk) or "LEAK" in msg.upper()
                return LegResult(leg.label, "LEAK" if is_leak else "ERROR", offer.id,
                                 f"{type(e).__name__}: {msg[:200]}")

    def _host_dead(self, host_id: str) -> str | None:
        """The provider's `dead_reason` for `host_id` if it exposes one and the
        host has visibly died, else None. Provider-agnostic: a provider without
        `dead_reason` (or a transient read) yields None = 'not confirmed dead', so
        a mid-run rc!=0 stays a work failure rather than a spurious failover."""
        dr = getattr(self.provider, "dead_reason", None)
        if dr is None:                                    # pragma: no cover
            return None
        try:
            return dr(host_id)
        except Exception:                                 # noqa: BLE001  # pragma: no cover
            return None

    # -- live-rental registry (signal-safe teardown backstop, #24) -----------
    def _track(self, instance_id: str) -> None:
        with self._live_lock:
            self._live[instance_id] = True

    def _untrack(self, instance_id: str) -> None:
        with self._live_lock:
            self._live.pop(instance_id, None)

    def _destroy_live(self) -> None:
        """Force-destroy every in-flight rental. Each `rent()` already tears down
        on its own exit; this is the backstop for a signal that would otherwise
        kill the process before those finallys run."""
        destroy = getattr(self.provider, "destroy", None)
        if destroy is None:
            return
        # RentedHost.id is a provider-opaque string in the Provider contract, so
        # pass it through as-is (no int cast) and keep the manual-cleanup hint
        # provider-agnostic -- this backstop must not assume a Vast id or CLI.
        name = getattr(self.provider, "name", "provider")
        with self._live_lock:
            ids = list(self._live)
        for iid in ids:
            try:
                destroy(iid)
                self._log(f"  signal teardown: destroyed {iid}")
            except Exception as e:                        # noqa: BLE001
                self._log(f"  signal teardown: FAILED to destroy {iid} ({e}) "
                          f"-- destroy instance {iid} manually via the {name} provider")

    # -- the run -------------------------------------------------------------
    def run(self, legs: Iterable[FleetLeg]) -> list[LegResult]:
        """Run every leg (parallel, one host each), failing over bad hosts and
        skipping already-complete legs. Returns one `LegResult` per input leg."""
        legs = list(legs)
        if not legs:
            return []
        labels = [leg.label for leg in legs]
        if len(set(labels)) != len(labels):
            raise ValueError("FleetLeg labels must be unique (they key output "
                             "dirs and the resume marker)")

        results: dict[str, LegResult] = {}
        pending: list[FleetLeg] = []
        for leg in legs:
            if self._complete(leg):                       # resume/skip (#26)
                results[leg.label] = LegResult(leg.label, "SKIP", None,
                                               "output already present")
            else:
                pending.append(leg)
        if pending:
            self._log(f"{len(pending)} leg(s) to run, {len(results)} skipped "
                      f"(already complete); up to {self.max_parallel} parallel")
            pool = _OfferPool(self.provider, self.host_spec,
                              max_refills=self.max_refills)
            with self._signal_guard():
                with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as ex:
                    futs = {ex.submit(self._run_leg, pool, leg, i): leg
                            for i, leg in enumerate(pending)}
                    for fut in cf.as_completed(futs):
                        r = fut.result()
                        results[r.label] = r
                        self._log(f"LEG {r.label}: {r.status}"
                                  + (f" ({r.detail})" if r.detail else ""))
        # Post-run reconciliation (#48): every tracked rental is untracked in its
        # leg's finally, so a non-empty _live here means a teardown gap (a rental
        # that escaped cleanup). Surface it loudly and sweep before it bills.
        with self._live_lock:
            stranded = list(self._live)
        if stranded:
            self._log(f"!! {len(stranded)} rental(s) STILL TRACKED after run "
                      f"(teardown gap) -- destroying: {stranded}")
            self._destroy_live()
        return [results[leg.label] for leg in legs]

    def _signal_guard(self):
        """For the duration of `run()`, make SIGTERM/SIGINT tear down live
        rentals before unwinding -- the proactive half of orphan prevention
        (#24), a backstop to each `rent()`'s own best-effort teardown."""
        return _SignalGuard(self)


class _SignalGuard:
    """Context manager that, for its lifetime, makes SIGTERM/SIGINT tear down the
    executor's live rentals before re-raising. Restores prior handlers on exit.
    Only the MAIN thread can install signal handlers, so a guard requested from a
    worker thread is a no-op (the run() that owns the threads installs it)."""

    _SIGNALS = (signal.SIGTERM, signal.SIGINT)

    def __init__(self, executor: FleetExecutor):
        self._exec = executor
        self._prev: dict[int, object] = {}

    def __enter__(self):
        if threading.current_thread() is not threading.main_thread():  # pragma: no cover
            return self                                   # can't set handlers off-main
        for sig in self._SIGNALS:
            try:
                self._prev[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle)
            except (ValueError, OSError):                 # pragma: no cover
                self._prev.pop(sig, None)                 # not settable here
        return self

    def _handle(self, signum, frame):
        self._exec._log(f"signal {signum}: tearing down live rentals")
        self._exec._destroy_live()
        prev = self._prev.get(signum)
        if callable(prev):
            prev(signum, frame)                           # chain prior handler
        else:
            raise KeyboardInterrupt                       # default: unwind the run

    def __exit__(self, *exc):
        for sig, prev in self._prev.items():
            if prev is None:                              # was set from C, not Python
                continue                                  # can't restore via signal()
            try:
                signal.signal(sig, prev)                  # restore
            except (ValueError, OSError, TypeError):      # pragma: no cover
                pass
        return False
