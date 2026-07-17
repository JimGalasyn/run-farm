"""Thin reference implementations of the campaign contract (design: CAMPAIGN.md).

Local-machine implementations of A/B/C/E that actually work, plus a
`SkyPilotExecutor` stub that documents the intended D mapping. These are the
proof that the protocols in `protocols.py` close -- not the production fleet
layer. The registry deliberately reuses `runs.py` (already physics-agnostic);
at extraction those helpers move into the standalone package.
"""

from __future__ import annotations

import json
from pathlib import Path

from run_farm.protocols import (
    Admission,
    AdmissionError,
    EventSink,
    HostReport,
    RunConfig,
    RunHandle,
    State,
)
from run_farm.config import (
    load_checkpoint,
    run_dir,
    save_checkpoint,
)


# ---------------------------------------------------------------- A + B (P4) --
class FileRunRegistry:
    """RunRegistry over a base directory, reusing `runs.py` for A and B.

    `register` -> config-hashed dir + MANIFEST.jsonl line (A). `save`/`load`
    are full-integrator-state checkpoints (B). Completion is a `DONE` marker
    carrying the result, so `is_complete` is the idempotent skip that makes
    spot preemption a no-op on finished runs.
    """

    def __init__(self, base: str | Path):
        self.base = Path(base)

    def register(self, config: RunConfig) -> RunHandle:
        d = run_dir(self.base, config)               # writes config.json + manifest
        return RunHandle(config=config, dir=d, name=config.run_name())

    def _ckpt(self, handle: RunHandle) -> Path:
        return handle.dir / "checkpoint.npz"

    def _done(self, handle: RunHandle) -> Path:
        return handle.dir / "DONE.json"

    def is_complete(self, handle: RunHandle) -> bool:
        return self._done(handle).exists()

    def load(self, handle: RunHandle) -> tuple[State, int] | None:
        p = self._ckpt(handle)
        if not p.exists():
            return None
        # Rebuild with the HANDLE's own config type, not the default: an engine
        # with its own RunConfig shape (model/N/L, grid_size/verts_per_side) would
        # make the default `SimpleRunConfig.from_json` choke on its fields. We
        # discard the rebuilt config anyway (the caller has `handle.config`), but
        # load_checkpoint reconstructs it, so it must reconstruct the right type.
        state, _config, step = load_checkpoint(p, config_class=type(handle.config))
        return state, step

    def save(self, handle: RunHandle, state: State, step: int) -> None:
        save_checkpoint(self._ckpt(handle), state, handle.config, step)

    def finish(self, handle: RunHandle, result: dict) -> None:
        self._done(handle).write_text(json.dumps(result, sort_keys=True) + "\n")


# -------------------------------------------------------------- C (P6, P7) --
class JsonlEventSink:
    """EventSink writing one `events.jsonl` per run, full fields under `triggered/`.

    `emit` streams small records (P6); `trigger` is the rare full-state capture
    for a flagged event -- the caller is responsible for having quenched first
    (P7), this layer only stores what it is handed.
    """

    def emit(self, handle: RunHandle, record: dict) -> None:
        with (handle.dir / "events.jsonl").open("a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def trigger(self, handle: RunHandle, state: State, reason: str) -> None:
        import numpy as np

        tdir = handle.dir / "triggered"
        tdir.mkdir(exist_ok=True)
        n = len(list(tdir.glob("*.npz")))
        arrays = {k: np.asarray(v) for k, v in state.items()}
        np.savez_compressed(tdir / f"event_{n:04d}.npz", __reason__=reason, **arrays)

    def close(self, handle: RunHandle) -> None:
        pass  # line-buffered append; nothing to flush in the file backend


# ---------------------------------------------------------------- E (P9) --
class ProbeAdmission:
    """Reference Admission: probe GPU presence, free memory, outbound bandwidth,
    and bail loudly below thresholds (P9).

    The probes here are structurally real but minimal -- the production version
    measures an actual outbound transfer and a real device query. The point is
    the SHAPE: measure, write what was measured, refuse on failure. Operational
    caveat: until `_outbound_mbps` times a real upload it returns +inf, so the
    LIVE guard today is effectively a GPU-presence + free-memory gate; the
    zero-outbound rejection (E's motivating failure) is exercised only by the
    mocked test. Wiring the real transfer is what makes E load-bearing.
    """

    def __init__(self, *, min_mem_gb: float = 4.0, min_mbps: float = 1.0,
                 require_gpu: bool = True):
        self.min_mem_gb = min_mem_gb
        self.min_mbps = min_mbps
        self.require_gpu = require_gpu   # fleet default True; False to run on a laptop

    def probe(self) -> HostReport:
        has_gpu, name, free_gb, probe_ok = self._device()
        return HostReport(
            has_gpu=has_gpu,
            device_name=name,
            free_mem_gb=free_gb,
            outbound_mbps=self._outbound_mbps(),
            probe_ok=probe_ok,
        )

    def guard(self) -> HostReport:
        r = self.probe()
        if not r.probe_ok:
            # A probe that couldn't run is a hard reject regardless of
            # require_gpu -- "runs anyway" on an unprobed host is the P9 sin.
            raise AdmissionError(f"device probe failed ({r.device_name})")
        if self.require_gpu and not r.has_gpu:
            raise AdmissionError(f"no GPU on host ({r.device_name})")
        if r.has_gpu and r.free_mem_gb < self.min_mem_gb:
            raise AdmissionError(
                f"insufficient device memory: {r.free_mem_gb:.1f} < "
                f"{self.min_mem_gb} GB")
        if r.outbound_mbps < self.min_mbps:
            raise AdmissionError(
                f"host cannot ship results: {r.outbound_mbps:.2f} < "
                f"{self.min_mbps} Mbps outbound (the standing failure mode)")
        return r

    @staticmethod
    def _device() -> tuple[bool, str, float, bool]:
        """Returns (has_gpu, device_name, free_gb, probe_ok)."""
        try:
            import jax
            d = jax.devices()[0]
            if d.platform != "gpu":
                return False, d.platform, 0.0, True   # probed fine, just no GPU
            stats = getattr(d, "memory_stats", lambda: {})() or {}
            limit = stats.get("bytes_limit", 0)
            used = stats.get("bytes_in_use", 0)
            # No memory_stats (limit==0) means UNKNOWN, not zero -- reporting 0.0
            # would falsely fail the mem gate on a healthy GPU. Don't block on a
            # capacity we couldn't measure.
            free_gb = (max(limit - used, 0) / 1e9) if limit else float("inf")
            return True, str(d.device_kind), free_gb, True
        except Exception as e:  # probing must never crash the worker -- it reports
            return False, f"probe-failed: {e}", 0.0, False   # -> hard reject

    @staticmethod
    def _outbound_mbps() -> float:
        # TODO: time a real upload to the result store. Placeholder keeps the
        # shape honest (a measured number flows into guard()), not the value.
        return float("inf")


# ---------------------------------------------------------------- D (adopted) --
class LocalExecutor:
    """Reference Executor: guard once, then run every task in-process.

    Stands in for the fleet so the contract is exercisable on a laptop. Spot
    recovery is trivially satisfied (nothing preempts a local run); the
    is_complete skip still applies if tasks are re-run.
    """

    def run(self, tasks, admission: Admission) -> None:
        admission.guard()                 # P9: probe-or-bail before any work
        for task in tasks:
            task()


class SkyPilotExecutor:
    """Adopt SkyPilot for D -- spot-fleet fan-out + preemption recovery (STUB).

    Intended mapping (the surveyed best-in-class for D; nothing to rebuild):
      - each task -> one SkyPilot *managed job* on a spot node, auto-recovered
        and cross-region/cross-cloud relaunched on preemption;
      - the node's `setup` phase runs `admission.guard()` (P9) before pulling
        work -- this is the piece SkyPilot itself does NOT provide;
      - the `RunRegistry` lives on a shared object store, so a relaunched job
        sees `is_complete`/`load` and resumes bit-identically instead of
        restarting from scratch (SkyPilot only re-runs from artifacts);
      - the work queue re-submits incomplete handles; finished ones no-op.

    Left unimplemented on purpose: wiring real `sky.jobs.launch` is fleet work
    (TODO.md collider-campaign item 4), built against THIS fixed contract.
    """

    def __init__(self, *, cloud: str | None = None, accelerators: str = "A10:1"):
        self.cloud = cloud
        self.accelerators = accelerators

    def run(self, tasks, admission: Admission) -> None:
        raise NotImplementedError(
            "SkyPilotExecutor is a contract stub; use LocalExecutor until the "
            "fleet layer lands (CAMPAIGN.md, TODO.md collider-campaign item 4).")
