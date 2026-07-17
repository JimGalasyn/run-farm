"""The campaign boundary: four physics-agnostic protocols (design: CAMPAIGN.md).

A campaign is 10^4-10^6 registered, restartable runs over a rented fleet. A
2026-06-12 literature sweep found NO library covers the full contract: the
executor layer (spot-fleet recovery) is solved by SkyPilot/dstack, but the
provenance/restart/event-record/admission contract is unserved -- and
host-probing admission (P9) exists nowhere. So this package owns A/B/C/E and
delegates D to a pluggable Executor.

Contract letters map to DESIGN.md principles:
  A  RunRegistry   config-hashed registry + idempotent skip          (P4)
  B  RunRegistry   full-integrator-state checkpoints, exact restart  (P4)
  C  EventSink     event-records-not-fields + triggered capture      (P6, P7)
  D  Executor      spot-fleet fan-out + preemption recovery          (adopted)
  E  Admission     probe-or-bail on flaky marketplace hosts          (P9)
  F  Provider      pluggable cloud broker: offers + leak-proof rent   (P9, P10)

A and B/C/E are the unserved contract this package owns; D is delegated to a
pluggable `Executor`. F is the second build-thin seam: the marketplace brokers
SkyPilot's providers can't drive (its Vast provider is broken against the live
API, P10) plug in here behind one Protocol, so a new cloud is a ~150-line
adapter, not a fork. The reference `VastProvider` lives in `vast.py`; serverless
backends (Modal, RunPod-serverless) have no host to rent and so are NOT
Providers -- they slot in at the Executor (D) seam instead.

The ONLY soliton-specific thing crossing this boundary is `RunFn`: the physics
is injected as a callable. No module under `campaign/` imports a model or
stepper, and the protocol + driver surface is jax-free; the one exception is the
reference `ProbeAdmission`, which imports jax lazily to probe the device. That
discipline -- no physics coupling -- is what keeps the layer extractable into a
standalone package at rule-of-three.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

# A run's full integrator state: an opaque flat dict of arrays (field, velocity,
# optimizer moments, RNG key, ...). The campaign layer never inspects it; only
# the injected physics (RunFn) understands its keys.
State = dict[str, Any]


@runtime_checkable
class RunConfig(Protocol):
    """A run's declarative identity: the six things the farm needs from a config.

    STRUCTURAL, not nominal -- bring your own dataclass. `SimpleRunConfig`
    (`run_farm.config`) is a batteries-included one; an engine with its own shape
    (a grid's N/L, a tissue's grid_size/verts_per_side, a world's n_cells) just
    satisfies this and keeps its own fields. The farm reads `.dtype` and `.params`
    and nothing else: everything else about a config is the engine's business.

    Note what is NOT required: **a seed**. Some configs are deterministic and seed
    one level up; seeding is a consumer's concern, and `seed_fn` in `run_farm.sweep`
    keeps it there.

    The contract that actually matters is that **`to_json` is canonical and
    stable**. `config_hash` IS the run's identity -- it names the run directory, and
    `RunRegistry.is_complete` resolves prior work by that name. Two processes
    serializing the same config must produce the same bytes, forever. Change your
    `to_json` and you rename every run you have ever written: `is_complete` stops
    recognizing finished work, a resumed campaign restarts from zero, and a rented
    fleet re-bills for results already on disk. Nothing raises. Pin it with a
    golden test.

    ⚠ **`from_json` is lossy in type, but NOT in hash.** JSON has no dataclass and
    no tuple, so any `params` value that isn't str/int/float/bool/list/dict returns
    as its JSON shadow -- a nested dataclass comes back a dict, a tuple comes back a
    list. The hash stays stable (so identity and idempotent-skip keep working, and
    nothing looks wrong), but a `RunFn` doing `params["rates"].point` or
    `params["range"].index(...)` **passes locally and breaks on the worker**, where
    the config was rebuilt through `from_json`. That local/remote divergence is the
    worst kind to debug on a rented box.

    So: **`params` values must be JSON-native, or your RunFn must accept the JSON
    shadow.** This is a documented contract edge, not a defect -- a typed codec
    would buy serialization complexity no consumer has asked for. Pinned in
    tests/test_config.py.

    Two gotchas on the Protocol itself:
      * `isinstance(cfg, RunConfig)` works (presence-only, per @runtime_checkable);
        **`issubclass()` raises TypeError** on a Protocol with data members. Don't.
      * `from_json` here is documentation, not a usable construction seam across a
        network: a worker holds bytes and no type. That is what the worker's
        `--config-class` ref is for (see `run_farm.remote.load_config_class`).
    """

    dtype: str
    """The run's float width. `remote.run_one` reads this -- and only this -- to
    enable x64 on a fresh worker before any array exists."""

    params: dict[str, Any]
    """Free-form per-run payload. The farm never interprets it, with one exception:
    `farm.py` keys its cut-flow on `params["rid"]`."""

    def to_json(self) -> str:
        """Canonical, stable serialization. See the class docstring: these bytes
        are the permanent names of your run directories."""
        ...

    @classmethod
    def from_json(cls, s: str) -> "RunConfig":
        """Rebuild from `to_json` bytes. Lossy in type (see class docstring)."""
        ...

    def config_hash(self, n: int = 12) -> str:
        """Stable short hash of `to_json` -- the run's identity (mechanism A)."""
        ...

    def run_name(self) -> str:
        """Human-legible directory name; must embed `config_hash` to stay unique."""
        ...


@dataclasses.dataclass(frozen=True)
class RunHandle:
    """Identity of one run: its config, its hashed directory, its name.

    Returned by `RunRegistry.register`; threaded through every capability so
    the registry, sink, and executor all agree on which run they touch.
    """

    config: RunConfig
    dir: Path
    name: str


@dataclasses.dataclass(frozen=True)
class HostReport:
    """What an `Admission` probe measured about a candidate host (P9).

    Hosts, networks, and devices lie; this is the measured truth a host is
    admitted or rejected on. Fields are deliberately concrete -- the standing
    case study is a 0.996-reliability host with ZERO outbound bandwidth.

    `probe_ok` is False when the probe itself failed (an exception, not a
    measured shortfall); such a host is a hard reject regardless of thresholds
    -- a probe that cannot run is not a host that may run work (P9).
    """

    has_gpu: bool
    device_name: str
    free_mem_gb: float
    outbound_mbps: float
    probe_ok: bool = True
    notes: str = ""


@dataclasses.dataclass
class RunContext:
    """The four orchestration capabilities handed to the physics, and nothing
    else. This dataclass IS the seam: `RunFn` receives exactly this.

      resume       prior full state to continue from, or None for a fresh run (B)
      resume_step  the step that `resume` was checkpointed at, or None if fresh
                   -- so a RunFn gets ledger/schedule continuity without having
                   to smuggle the counter through `State` (B)
      checkpoint   persist full integrator state at a step (B)
      emit         stream one small event record -- a ledger row, a census (C)
      trigger      capture full fields on a flagged event, after a quench (C/P7)
    """

    resume: State | None
    checkpoint: Callable[[State, int], None]
    emit: Callable[[dict], None]
    trigger: Callable[[State, str], None]
    resume_step: int | None = None


# The single soliton-specific injection. Returns a small result record (the
# run's summary ledger), never raw fields -- those go through ctx.trigger (P6).
RunFn = Callable[[RunConfig, RunContext], dict]


@runtime_checkable
class RunRegistry(Protocol):
    """A (config-hashed) registry of runs with full-state restart (A + B, P4).

    A result that cannot name its config hash does not exist. Restart is
    bit-identical at fixed dtype/devices because checkpoints carry FULL
    integrator state, not just artifacts.
    """

    def register(self, config: RunConfig) -> RunHandle:
        """Create/locate the config-hashed run dir; append one manifest line."""
        ...

    def is_complete(self, handle: RunHandle) -> bool:
        """True if this run already finished -- the idempotent-skip that makes
        spot preemption free (re-submitting a done run is a no-op)."""
        ...

    def load(self, handle: RunHandle) -> tuple[State, int] | None:
        """Latest full-state checkpoint as (state, step), or None if none yet."""
        ...

    def save(self, handle: RunHandle, state: State, step: int) -> None:
        """Write a full-state checkpoint (field + velocity/optimizer + RNG)."""
        ...

    def finish(self, handle: RunHandle, result: dict) -> None:
        """Mark the run complete and record its summary result."""
        ...


@runtime_checkable
class EventSink(Protocol):
    """Streaming event records, with full fields only on triggered events (C).

    At campaign scale the product of a run is a small record, not its fields
    (P6). Cheap classifiers stream via `emit` on every event; expensive
    full-state capture fires through `trigger`, and ONLY after a quench, since
    descent cannot create topology -- relax-then-ID is faithful, in-bath is not
    (P7).
    """

    def emit(self, handle: RunHandle, record: dict) -> None:
        """Append one small event record (a charge/energy ledger row, a census)."""
        ...

    def trigger(self, handle: RunHandle, state: State, reason: str) -> None:
        """Capture full fields for a flagged event (the rare, kept snapshot)."""
        ...

    def close(self, handle: RunHandle) -> None:
        """Flush this run's stream."""
        ...


@runtime_checkable
class Admission(Protocol):
    """Probe-or-bail admission control for flaky fleets (E, P9).

    Served by no existing orchestrator: they assume reliable hosts. Anything
    that touches infrastructure probes first, writes what it measured, and
    bails early -- never "runs anyway" on unverified capacity.
    """

    def probe(self) -> HostReport:
        """Measure this host's real compute + network capacity."""
        ...

    def guard(self) -> HostReport:
        """Probe and raise `AdmissionError` if the host fails the bar. Called on
        each fleet node before it pulls work. Returns the report on success."""
        ...


@runtime_checkable
class Executor(Protocol):
    """Fan tasks out over a fleet and recover from preemption (D -- adopted).

    The crowded, solved layer: adopt SkyPilot/dstack, never rebuild it. The
    executor's only campaign-specific duty is to guard each worker via
    `Admission` before it runs, and to lean on `RunRegistry.is_complete` so a
    preempted task is simply re-submitted and skips its finished work.
    """

    def run(self, tasks: Iterable[Callable[[], None]],
            admission: Admission) -> None:
        """Execute every task thunk across the fleet, guarding each worker.

        `tasks` is an Iterable, not a list: the driver passes a generator so
        neither the configs nor the thunks are materialized on the submitting
        node -- a 10^4-10^6-run queue streams rather than spiking memory.
        """
        ...


class AdmissionError(RuntimeError):
    """A host failed its capacity probe and must not run work (P9)."""


# ============================================================ F (P9, P10) ====
# The cloud-broker seam: pick a host from a marketplace, rent it, and ALWAYS
# tear it down. Selection metadata lies (P9), provider SDKs lag the live API
# (P10), and a leaked GPU bills by the second -- so the Protocol fixes the
# *shape* (offers -> rent -> guaranteed teardown) and a thin per-provider
# adapter fills in the HTTP. Only "rent-a-box-and-SSH" markets (Vast, RunPod
# pods, TensorDock, EC2 spot) are Providers; serverless is an Executor (D).


@dataclasses.dataclass(frozen=True)
class HostSpec:
    """What the campaign needs from a host -- how a `Provider` selects offers.

    The selection-time half of admission (P9): a coarse filter the marketplace
    can apply server-side, made rigorous per-host later by `Admission.guard`.

    `min_cuda` is load-bearing and easy to get wrong: it must be >= the launch
    image's CUDA floor (`NVIDIA_REQUIRE_CUDA`), or the nvidia-container OCI hook
    fails at *container create* on hosts whose driver is older than the image --
    a failure that looks like a broken host but is really an image/host mismatch
    (measured 2026-06-14: a cuda:12.4 image hard-failed create on ~16/17 cheap
    hosts; cuda:12.2 + min_cuda=12.2 -> 0 failures).
    """

    gpu_name: str = "RTX_3090"
    num_gpus: int = 1
    max_dph: float = 0.40            # $ per hour ceiling, all-in
    min_reliability: float = 0.95
    min_inet_mbps: float = 100.0
    min_cuda: float = 12.0           # >= the LaunchSpec image's CUDA floor
    min_gpu_frac: float = 0.0        # >0 gates for DEDICATED machines (1.0 = whole box,
                                     # no GPU-sharing tenants). Guards against the
                                     # oversubscribed-shared-host thrash (load-36) that
                                     # silently starves a long run. 0 = no gate (default).


@dataclasses.dataclass(frozen=True)
class Offer:
    """A rentable host a `Provider` surfaced (the fields an executor decides on).

    `id` is a provider-opaque string (not every market uses ints); `provider`
    names which adapter minted it so a mixed-provider queue stays attributable.
    """

    id: str
    dph: float                       # $ per hour, all-in
    gpu_name: str
    num_gpus: int
    reliability: float               # 0..1
    inet_down_mbps: float
    cuda_max: float                  # host's max supported CUDA (the P10 gate)
    geolocation: str = ""
    provider: str = ""

    def __str__(self) -> str:
        return (f"offer {self.id}: {self.num_gpus}x {self.gpu_name} "
                f"${self.dph:.3f}/hr  rel={self.reliability:.3f}  "
                f"down={self.inet_down_mbps:.0f}Mbps  cuda<={self.cuda_max}  "
                f"[{self.geolocation.strip(', ')}]"
                + (f"  @{self.provider}" if self.provider else ""))


@dataclasses.dataclass(frozen=True)
class LaunchSpec:
    """How to bring a rented host up: the container image + bootstrap + disk.

    `image`'s CUDA floor must be <= the chosen offer's `cuda_max` (see
    `HostSpec.min_cuda`). `onstart` is the bootstrap script the host runs on
    boot; it should signal readiness/failure so `rent` can probe-and-bail (P9).

    `label` stamps every instance the provider rents under this spec (e.g. a
    campaign/run id like ``"eps-kick-farm-2026-06-15"``) so a live instance is
    attributable to the run that created it -- a human at the console (or
    ``vastai show instances``) can then spot an orphan by its label. It is also
    the hook a future label-scoped `reap` would filter on; `reap` today still
    scopes by ledger-id diff or ``--all`` (#24).
    """

    image: str
    onstart: str
    disk_gb: float = 40.0
    label: str = "run-farm"


@dataclasses.dataclass(frozen=True)
class RentedHost:
    """A live, reachable host that `Provider.rent` yields.

    SSH coordinates because Providers are the rent-a-box markets; the executor
    ships inputs and a worker command here. `offer` is carried so cost and geo
    stay attached to the running host; `raw` is the provider's untyped record.
    """

    id: str
    ssh_host: str
    ssh_port: int
    offer: Offer
    raw: dict = dataclasses.field(default_factory=dict)


class HostProbeFailed(RuntimeError):
    """A rented host came up unusable -- DNS/image-pull/disk/zero-outbound: the
    P9 'hosts lie' case. The executor tears it down and fails over to the next
    offer rather than running work on it."""


class RentUnavailable(RuntimeError):
    """An offer could NOT be rented -- it was taken between `offers()` and
    `rent()` (a marketplace race), or the create call failed before any instance
    existed. Distinct from `HostProbeFailed` (a host that came up but is
    unusable) and from a leak (`rent` raises loudly on that): an unavailable
    offer never created an instance, so there is nothing to tear down and the
    executor simply fails over to the next offer. Providers raise this for the
    pre-instance create failure so an executor's failover path stays
    provider-agnostic (no need to string-match a provider-specific error)."""


class LeakRisk(RuntimeError):
    """A rented host could NOT be confirmed torn down -- destroy failed or the
    instance is still present after teardown -- so a GPU may still be billing.
    The provider-agnostic, structured leak signal: a Provider's `rent()` raises
    this (not a stringly-typed "LEAK" message) so a fleet/executor distinguishes a
    cost-safety alarm from an ordinary error by *type*, and a missed leak can't
    hide behind a reworded message. Unlike `RentUnavailable`/`HostProbeFailed`
    this is NOT a failover signal -- it must surface loudly, not be retried."""


@runtime_checkable
class Provider(Protocol):
    """A pluggable cloud broker: list offers, rent one, ALWAYS tear it down (F).

    The contract is three things, and the third is the point:

      1. `offers(spec)` returns rentable hosts meeting the bar, cheapest first.
      2. `rent(offer, launch)` brings a host up and yields a `RentedHost`.
      3. **`rent` is a context manager whose teardown is guaranteed and
         verified** -- it destroys the host on EVERY exit (success, exception,
         Ctrl-C) and independently confirms it is gone, raising on a leak. A
         leaked GPU bills by the second; teardown is the contract, not an
         implementation nicety. Adapters that cannot prove teardown do not
         satisfy `Provider`.

    `rent` raises `HostProbeFailed` when the host comes up unusable, so the
    executor can fail over; any other provider error propagates. `name`
    identifies the adapter (stamped onto `Offer.provider`).
    """

    name: str

    def offers(self, spec: HostSpec) -> list[Offer]:
        """Rentable offers meeting `spec`, cheapest first (free; no spend)."""
        ...

    def rent(self, offer: Offer, launch: LaunchSpec, *,
             timeout_s: float = 600) -> AbstractContextManager[RentedHost]:
        """Rent `offer`, wait until usable, yield a `RentedHost`, guarantee+verify
        teardown on exit. Raises `HostProbeFailed` if the host never comes up."""
        ...
