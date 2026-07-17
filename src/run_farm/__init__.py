"""run-farm: checkpointed campaign runs over a rented GPU spot fleet.

Physics-agnostic orchestration for 10^4-10^6 registered, restartable runs. Six
contracts (design note: docs/DESIGN.md):

  A/B  RunRegistry   config-hashed registry + full-state, restart-exact checkpoints
  C    EventSink     event-records-not-fields + triggered full capture
  D    Executor      spot-fleet fan-out + preemption recovery (adopt SkyPilot/dstack)
  E    Admission     probe-or-bail on flaky marketplace hosts
  F    Provider      pluggable cloud broker: offers + best-effort verified teardown

A 2026-06-12 literature sweep confirmed no library covers A/B/C/E -- host-probing
admission (E) exists nowhere. The only thing an engine injects is a `RunFn`
(`Callable[[RunConfig, RunContext], dict]`); no module here imports a model or a
solver, and the surface is jax-free apart from checkpoint array I/O. Bring your own
config (satisfy the `RunConfig` Protocol) and your own RunFn; the farm does the rest.
"""

from run_farm.driver import execute_config, run_campaign
from run_farm.protocols import (
    Admission,
    AdmissionError,
    EventSink,
    Executor,
    HostProbeFailed,
    HostReport,
    HostSpec,
    LaunchSpec,
    Offer,
    Provider,
    RentedHost,
    RentUnavailable,
    RunContext,
    RunFn,
    RunHandle,
    RunRegistry,
    State,
)
from run_farm.reference import (
    FileRunRegistry,
    JsonlEventSink,
    LocalExecutor,
    ProbeAdmission,
    SkyPilotExecutor,
)
from run_farm.multi import (
    CampaignReport,
    run_multi,
    split_configs,
    stream_multi,
)
from run_farm.local_exec import InProcessExecutor
from run_farm.provider_exec import ProviderExecutor
from run_farm.fleet import (
    FleetExecutor,
    FleetLeg,
    ImportReady,
    LegResult,
    SentinelReady,
    parse_progress_line,
)
from run_farm.status import fleet_status
from run_farm.remote import load_run_fn, run_one
from run_farm.store import (
    BlobStore,
    MemoryBlobStore,
    ObjectStoreEventSink,
    ObjectStoreRunRegistry,
    S3BlobStore,
)
from run_farm.runpod import RunPodProvider
from run_farm.vast import VastLedger, VastProvider
from run_farm.ledger import RentalLedger
from run_farm.config import (
    SimpleRunConfig,
    load_checkpoint,
    run_dir,
    save_checkpoint,
)
from run_farm.protocols import RunConfig
from run_farm.sweep import legs
from run_farm.budget import BudgetExceeded, CappedProvider, estimate
from run_farm import testing
from run_farm.farm import (
    CutFlow,
    FarmCampaign,
    launch_gate,
    leg_params,
    simple_leg_to_config,
    verify_shipment,
)

# ModalExecutor is intentionally NOT imported here: `modal` is an optional
# dependency, so `import run_farm` must not require it. Use
# `from run_farm.modal_exec import ModalExecutor` when modal is
# installed.

__all__ = [
    # config: the RunConfig Protocol + a batteries-included one + A/B helpers
    "RunConfig", "SimpleRunConfig",
    "save_checkpoint", "load_checkpoint", "run_dir",
    # protocols (the contract)
    "RunRegistry", "EventSink", "Admission", "Executor", "Provider",
    "RunContext", "RunFn", "RunHandle", "HostReport", "State", "AdmissionError",
    # F (cloud broker) contract types
    "HostSpec", "LaunchSpec", "Offer", "RentedHost", "HostProbeFailed",
    "RentUnavailable",
    # reference implementations
    "FileRunRegistry", "JsonlEventSink", "ProbeAdmission",
    "LocalExecutor", "SkyPilotExecutor", "ProviderExecutor", "InProcessExecutor",
    "VastProvider", "VastLedger", "RentalLedger", "RunPodProvider",
    # parallel script-fleet executor (#25) + live status
    "FleetExecutor", "FleetLeg", "LegResult", "ImportReady", "SentinelReady",
    "parse_progress_line",
    "fleet_status",
    # shared object-store backend (A/B/C over a BlobStore)
    "BlobStore", "MemoryBlobStore", "S3BlobStore",
    "ObjectStoreRunRegistry", "ObjectStoreEventSink",
    # driver + remote-worker core
    "run_campaign", "execute_config", "run_one", "load_run_fn",
    # multi-provider partition-and-merge
    "run_multi", "stream_multi", "split_configs", "CampaignReport",
    # campaign axes (arm x replicate x grid) + cost governance
    "legs", "estimate", "CappedProvider", "BudgetExceeded",
    # physics-free RunFns for smoke-testing a fleet with no engine
    "testing",
    # governed campaign: policy over mechanism (preflight, launch gate,
    # shipment/SHA verification, cut-flow, ingest) — domain policy injected
    "FarmCampaign", "CutFlow", "launch_gate", "verify_shipment",
    "leg_params", "simple_leg_to_config",
]
