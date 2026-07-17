# The run-farm design

A *campaign* is 10⁴–10⁶ registered, restartable runs over a rented fleet. This
note is the design of the boundary between an **engine** (the physics/simulation)
and the **orchestration** (this package). run-farm owns the orchestration and
nothing else: an engine injects a single `RunFn` and its own config shape, and no
module here imports a model or a solver.

This document was written while the layer lived inside
[jax-solitons](https://github.com/JimGalasyn/jax-solitons) as an internal
`campaign/` module; it was extracted to this standalone package in 2026-07 (see
*Extraction*, below). Where it says "the physics", read "the engine you inject".

## Why a boundary at all (the research, 2026-06-12)

Two verified literature sweeps drive this:

1. **The niche is open.** No JAX soliton/topological-field engine and no
   farm-scale soliton campaign system exist in the verified literature
   (10 candidate URLs, 0 qualified). Campaign-scale orchestration *is* the
   moat — see README positioning.
2. **Build thin, don't rebuild.** Of 8 orchestration tools assessed against a
   five-part contract, **none covers it**, but the split is uneven:

   | Contract | What it is | Best off-the-shelf | Verdict |
   |---|---|---|---|
   | **A** | config-hashed run registry | DVC (partial, stage-level) | **build** (already have it) |
   | **B** | bit-identical full-integrator-state restart | — (all "re-run from artifact") | **build** (already have it) |
   | **C** | event-records-not-fields + triggered full-state capture | Balsam/Parsl (event logs, no trigger) | **build** |
   | **D** | spot-fleet fan-out + preemption recovery | **SkyPilot / dstack** (clean) | **adopt** |
   | **E** | probe-or-bail admission on flaky hosts | — (served by no one) | **build** (novel) |

   The executor layer (D) is a crowded, solved market — adopt SkyPilot, never
   rebuild it. The provenance/restart/event/admission contract (A/B/C/E) is
   unserved, and **E exists nowhere** — every surveyed tool assumes reliable
   hosts. That is the part worth owning.

   **Update (2026-06-13), measured the hard way (P10):** SkyPilot 0.12.x's Vast
   provider — and the official `vastai` SDK — are broken against Vast's live
   API (the bare `GET /api/v0/instances/` collection returns HTTP 410 Gone;
   both route instance-listing there). Because D is *pluggable*, we dropped a
   thin stdlib `VastClient` (`campaign/vast.py`) — direct to the endpoints that
   work (v1 for listing, v0 sub-resources for create/destroy/logs), no SkyPilot,
   no SDK. (The `VastExecutor` that wires it into `run_campaign` is the next
   step; this PR lands the client + `rent()` lifecycle.) A live run
   validated it end-to-end (search → create → run the GPU tier → destroy, with
   host fail-over and a verified-clean teardown). The lesson reinforces, not
   contradicts, "adopt, don't rebuild": we still adopt SkyPilot where its
   providers work; for one marketplace whose provider is broken, a thin direct
   client is the build-thin move. And the dead-DNS host that the run failed
   over (0.99-reliability on paper, unreachable in fact) is **E/P9 demonstrated
   live** — caught by `HostProbeFailed`, logged to the `VastLedger`.

## The F seam: `Provider`, and why serverless isn't one (2026-06-14)

The `VastClient` started as one marketplace's thin client. It is now the
reference implementation of a **`Provider` Protocol (F)** — `offers(HostSpec)`
plus a teardown-verifying `rent(Offer, LaunchSpec)` — so a new cloud (RunPod pods,
TensorDock, EC2 spot) is a ~150-line adapter against a fixed contract, not a
fork of the orchestration. This is the same "build thin" call as D, applied
once more: SkyPilot *adopts* the clouds whose providers work; F *owns* the thin
broker for the marketplaces whose providers are broken (P10).

Two things make F a real contract and not just an interface:

- **Best-effort verified teardown is the invariant, not a nicety.** `rent` is a context
  manager that destroys the host on *every* exit and independently verifies it
  is gone, raising on a confirmed leak. An adapter that can't prove teardown
  doesn't satisfy `Provider`. The contract test drives this with a
  `FakeProvider` (no spend): teardown fires on success, on exception, and on a
  `HostProbeFailed` host, and the cheapest-first failover idiom leaves nothing
  billing.
- **`HostSpec.min_cuda` is load-bearing (P10).** It must be ≥ the launch
  image's CUDA floor, or the nvidia-container OCI hook fails at *container
  create* on any host whose driver is older than the image — a failure that
  reads as "broken host" but is an image/host mismatch. Measured 2026-06-14: a
  `cuda:12.4` image hard-failed create on ~16/17 cheap hosts; `cuda:12.2` +
  `min_cuda=12.2` → 0 failures. The gate now lives in the offer query.

**Serverless backends are not Providers.** Modal and RunPod-serverless have no
host to rent, no SSH, no teardown to own — you submit a containerized function
and the platform provisions. They therefore plug in at the **Executor (D)**
seam (a future `ModalExecutor`), not F. F is exactly the rent-a-box-and-SSH
markets; the line is "is there a host whose meter *I* must turn off?".

**VM marketplaces are out of scope (evaluated 2026-06-14).** TensorDock is a
rent-a-box market that *would* be a Provider — except it deploys **VMs, not
containers**: `LaunchSpec.image` would be an OS slug (`ubuntu2404`) not a docker
ref, there's no container entrypoint, and bootstrap must run over SSH post-boot.
Rather than overload `LaunchSpec` with two meanings (docker ref vs OS image) and
two bootstrap paths, the Provider seam stays **container-based** (docker image +
onstart entrypoint) — the shape Vast and RunPod share. A VM marketplace is
deferred until there's a clear need; if one lands, it argues for a distinct
VM-provider type, not a broadened `LaunchSpec`.

**Second adapter — the rule-of-three test (2026-06-14).** `RunPodProvider`
(`runpod.py`) is a second cloud behind the same seam, and it held — but it
flexed the Protocol in exactly the places a one-marketplace abstraction would
have hard-coded, which is the point of building it:

- **Offers are GPU *types*, not hosts.** RunPod's REST API is pod-centric with
  no host catalog; the type list + pricing is in its *GraphQL* API. So
  `offers()` returns one synthetic `Offer` per (type, cloud-tier), `id` = the
  gpuTypeId, and `rent()` asks RunPod to *place* a pod rather than pick a named
  machine. Per-host reliability/inet aren't in the catalog → NaN (honest
  "unknown"), not a faked number.
- **Admission splits across `offers`/`rent`.** Vast filters everything at
  selection; RunPod resolves the CUDA floor and bandwidth at pod *create*
  (`allowedCudaVersions`, `minDownloadMbps`). So `HostSpec.min_cuda` is honored
  at rent for RunPod, at selection for Vast — same intent, different seam half.
- **Cloudflare UA footgun.** RunPod's API 403s the default `Python-urllib` User-
  Agent (Cloudflare error 1010); any explicit UA clears it. Vast had no such
  filter. Encoded in `runpod._req`.

The shared `Offer`/`RentedHost`/`HostSpec`/`LaunchSpec` types and the teardown-verifying
`rent` contract needed **zero changes** to absorb all three — the seam was drawn
right. Both adapters are stdlib-only (urllib) and CI-tested against a mocked
HTTP layer (no spend); `offers()` was also validated live against each cloud's
real catalog.

This lands the F contract + **two** reference adapters + their tests; the
executors that drive them are below.

## Remote execution: shipping work, not closures (2026-06-14)

`Executor.run(tasks, admission)` takes opaque thunks that close over the local
`run_fn` and registry — a fine model for an **in-process** worker (`LocalExecutor`)
or a SkyPilot-managed node that re-runs the program, but a closure can't cross a
network. So remote execution ships three serializable things instead: the
**configs** (`RunConfig.to_json`), the **RunFn by name** (a `'module:function'`
ref the worker imports — the seam's one physics injection, made shippable), and
a **work dir** on shared storage. `campaign.remote.run_one` is the unit both
remote workers run; it calls the same `execute_config` (factored out of the
driver) so register/skip/resume/finish semantics are identical on every machine.

Two remote executors consume this, one per cloud topology:

- **`ModalExecutor`** (serverless / D). `Function.map` over the configs; each
  call runs `run_one` against a **Modal Volume** mounted as the registry root
  (checkpoints/events/triggered captures persist there; small result records
  ride the `.map` return). No host to rent or leak — Modal owns the lifecycle —
  so `admission` (E) is a no-op. `modal` is an optional dependency: imported only
  by `campaign.modal_exec`, never by `campaign/__init__`.
- **`ProviderExecutor`** (rented fleet / D over F). Drives any `Provider`: pull
  offers, rent with **per-host failover** (`HostProbeFailed` → next offer), wait
  for the engine to come up (probe `import jax_solitons` over SSH, P9), run the
  `worker` CLI per config over SSH, sync artifacts back, and lean on the
  Provider's teardown-verifying `rent()` for teardown. The principled generalization of
  the hand-rolled `run_eps_fleet` driver. (v1 is single-host-sequential;
  multi-host parallel fan-out is a follow-up.)

**Across providers (partition-and-merge).** `multi.run_multi` drives one
campaign over several executors at once -- `split_configs` partitions the work
(round-robin or weighted toward cheaper clouds), each executor runs its slice
concurrently, and the result records merge into one harvest. This is *sensible*
because run identity is content-addressed (`config_hash`): a record names its
config regardless of which cloud produced it, so the merge can't collide, and a
failed provider is isolated (its slice errors; the rest still harvest).
`stream_multi` yields each provider's slice the instant it completes (a fast
cloud isn't gated on the slowest); `run_multi(on_result=...)` observes them live.
The partition is assumed **disjoint** -- `is_complete`/resume are per-executor
*when each writes its own local store*.

**Shared object-store backend (`campaign.store`).** Point every executor at one
store and that caveat lifts: `ObjectStoreRunRegistry` (A/B) + `ObjectStoreEventSink`
(C) implement the existing protocols over a minimal `BlobStore` (`MemoryBlobStore`;
`S3BlobStore` for S3/R2/GCS/MinIO, `boto3` optional). One shared store is the
single source of truth -- global `is_complete` (dedup + cross-provider and
cross-restart resume) and one place to read both results and the *streamed* event
ledger (`events(name)`), the real-time cross-provider view. Every record is its
own blob (object stores have no atomic append), so concurrent writers -- the
whole point -- never contend. A second backend of A/B/C, not a new seam. The
remaining wiring (a rented box writing straight to the store needs scoped creds;
the conservative path is the orchestrator holding the shared registry, checking
`is_complete` before dispatch and writing results back) is the integration tier
on top of this.

## The contract → DESIGN.md principles

| Letter | Protocol | Responsibility | Principle |
|---|---|---|---|
| A, B | `RunRegistry` | config-hashed dirs + manifest; full-state checkpoints; idempotent skip | P4 |
| C | `EventSink` | stream small per-run records; capture full fields only when triggered | P6, P7 |
| D | `Executor` | fan out tasks over a fleet; recover from preemption (re-submit incomplete) | adopted |
| E | `Admission` | probe a host's real compute+network capacity; bail loudly on bad hosts | P9 |
| F | `Provider` | pluggable cloud broker: list offers, rent one, **best-effort verified teardown** | P9, P10 |

A/B already live in `runs.py` (`RunConfig.config_hash`, `save_checkpoint`,
`load_checkpoint`) — they are *already physics-agnostic*. The campaign module
wraps them today; at extraction time they move wholesale into the new package.

## The one seam the physics crosses

The **only** soliton-specific thing that crosses this boundary is a single
injected callable:

```python
RunFn = Callable[[RunConfig, RunContext], dict]
```

`RunContext` hands the physics four orchestration capabilities and nothing
else: `ctx.resume` (prior full state or `None`), `ctx.checkpoint(state, step)`,
`ctx.emit(record)`, `ctx.trigger(state, reason)`. The physics returns a small
result record. **No module under `campaign/` imports a model or a stepper**,
and the protocol + driver surface is jax-free (beyond the array-I/O already in
`runs.py`); the lone jax touch is the reference `ProbeAdmission`, which imports
jax lazily to query the device. That no-physics-coupling discipline is what
keeps the layer extractable.

## Extraction (done, 2026-07)

The rule-of-three gate this document originally set was: keep the layer internal
until **either** a second real consumer appears **or** the A/B/C/E API stabilizes
through one real campaign. The **second** disjunct was met — the Provider (F) seam
absorbed three backends (Vast, RunPod, Modal) with zero Protocol changes, and C/E
shipped and are CI-gated. jax-morpho's evolution loop is the credible near-future
second consumer; the extraction is what lets it share the orchestration without
coupling to soliton physics (its config is genome/selection-shaped, not N/L). So
the layer was lifted here, `RunConfig`/checkpoint helpers and all.

## Status

`protocols.py` is the contract; `reference.py` has thin local-machine
implementations (functional) plus a `SkyPilotExecutor` stub (documents the
intended mapping, raises `NotImplementedError`); `vast.py` is the reference
`Provider` (F). The local/reference path is **CI-gated** — the A/B/C/E contract is
driven end to end over the physics-free RunFns in `run_farm.testing` (register →
checkpoint-per-step → resume, bit-identical); the F contract is tested via a
zero-spend `FakeProvider` (best-effort teardown on success/exception/bad-host +
failover); the shared remote core (`run_one`/`load_run_fn`) and `ProviderExecutor`
(failover + teardown over a `FakeProvider`, mocked SSH) are CI-gated too. The
`ModalExecutor` is validated by a live run (not CI — it needs Modal + a GPU), and
the F contract has been validated live against Vast/RunPod (offers, rent, verified
teardown, 0 leaks). Only the `SkyPilotExecutor` remains a stub.
