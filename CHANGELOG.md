# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project follows
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features).

## [Unreleased]

## [0.1.0] — Extraction from jax-solitons

The initial release. run-farm is the campaign/GPU-farming layer extracted from
[jax-solitons](https://github.com/JimGalasyn/jax-solitons), where it lived as an
internal `campaign/` module through one real collider campaign. It comes out
because the A/B/C/E API stabilized: the Provider (F) seam absorbed three backends
(Vast, RunPod, Modal) with zero Protocol changes, and jax-morpho's evolution loop
is a credible near-future second consumer — the extraction is what lets it share
the orchestration without coupling to soliton physics.

### Added
- **The six-contract campaign boundary** (`protocols.py`): `RunRegistry` (A/B),
  `EventSink` (C), `Executor` (D), `Admission` (E), `Provider` (F), plus a
  physics-blind `run_campaign` driver. The only thing an engine injects is a
  `RunFn` (`Callable[[RunConfig, RunContext], dict]`).
- **`RunConfig` as a structural Protocol** plus a batteries-included
  `SimpleRunConfig`. An engine brings its own config shape (a grid's N/L, a
  tissue's grid_size/verts_per_side); the farm reads only `.dtype` and `.params`.
  Config identity is a stable content hash that names the run directory.
- **Restart-exact registry + full-state checkpoints** (`config.py`): a preempted
  run resumes bit-identically; re-submitting a finished run is a no-op.
- **Probe-or-bail admission** (`ProbeAdmission`) and **leak-proof cloud brokers**
  (`VastProvider`, `RunPodProvider`): every `rent()` destroys its host on every
  exit and independently verifies teardown, raising on a leak.
- **Executors**: `LocalExecutor`/`InProcessExecutor` (in-process),
  `ProviderExecutor` and `FleetExecutor` (rent-a-box with per-host failover),
  `ModalExecutor` (serverless; needs the `modal` extra).
- **Shared object-store backend** (`store.py`): `ObjectStoreRunRegistry` /
  `ObjectStoreEventSink` over a `BlobStore`, for global dedup and cross-cloud
  resume (S3 backend needs the `s3` extra).
- **Campaign axes** (`sweep.py`): `legs()` expands (arm × replicate × grid) into a
  campaign, with caller-supplied `seed_fn` so seed provenance stays with the
  consumer. Arm and replicate are first-class so replicate-lineage experiments
  (and a built-in calibration arm) group cleanly off the result records.
- **Cost as a first-class quantity** (`budget.py`): `estimate()` gives a
  pre-launch dollar number including the failure tax (you pay for hosts that never
  boot); `CappedProvider` is a Provider decorator that *refuses* to rent past a
  dollar cap, counting booked spend plus in-flight burn.
- **`RentalLedger`** (`ledger.py`): append-only rental receipts (spend + outcomes),
  provider-agnostic — it is also the complete record of created instances, so an
  orphan sweep that consults it closes the create→track leak window.
- **Physics-free RunFns** (`testing.py`): `echo_run_fn`, `counting_run_fn`,
  `failing_run_fn` — importable on a rented box, so you can smoke-test a real fleet
  end to end for pennies before pointing an expensive engine at it.

### Notes
- The Vast broker is deliberately stdlib-only (the vastai SDK breaks against the
  live API), so there is no `vast` extra.
- The orchestration surface is jax-free apart from checkpoint array I/O; `jax` is a
  base dependency only for `.npz` state and the lazy device probe.
