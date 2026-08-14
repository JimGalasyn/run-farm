# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project follows
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features).

## [Unreleased]

### Fixed
- **CI was red on `main` because nine RunPod tests needed a key only the dev machine has**
  (`tests/test_runpod_provider.py`). `RunPodProvider.pubkey_path` defaults to
  `~/.ssh/vastai.pub` and `create` reads it to fill `env.PUBLIC_KEY`, so every test
  reaching `create` depended on that file *existing on the developer's machine*. Green
  locally, nine failures on any runner, surfacing as `FileNotFoundError` inside `_pubkey`
  rather than as anything about the behaviour under test. `main`'s last two CI runs
  (`2e2b514`, `521d106`) failed identically, so every PR was landing on a red baseline —
  which is the expensive part: a suite that is always red cannot tell you that you broke
  something. The `mk` fixture now defaults `pubkey_path` to a real file under `tmp_path`,
  via `setdefault` so a future `create` test cannot reintroduce the dependency by
  forgetting it, while tests *about* key resolution still override it and exercise the
  real lookup. `test_create_defaults_to_the_executors_own_key` is built directly instead,
  since its subject is the default and a fixture supplying one would assert the fixture.
  Verified by hiding `$HOME`: 377 passed with and without it, against 9 failures before.

### Added
- **The gauntlet reaches the registry path** (`gauntlet.py`). Everything in that module
  served the FLEET path — rent hosts, ship a payload, run `FleetLeg`s, resume off a
  `done_when` marker. `run_campaign` + a `RunRegistry` is the library's other first-class
  path and had **no gauntlet coverage at all**, which meant `ResumeMarkersIntended` —
  the check whose whole purpose is *a skip is a claim that work is already done and
  deserves the same scrutiny as a result* — could not be run over half the library. The
  failure is identical on both paths; only the marker differs.
  - **`RegistryMarkersIntended`**: names every config the registry already considers
    complete, with its age. **Read-only, deliberately**: it does not call
    `register`, because that writes a run dir and a manifest line, and `run_campaign`
    defers both to the worker on purpose ("at 10^4–10^6 scale an eager
    `[register(c) for c in configs]` would serialize that many mkdir + manifest appends
    on one node"). A pre-registering check would reintroduce exactly that cost and
    litter the output directory with runs that never happen. It builds a `RunHandle`
    from `config.run_name()` — protocol-guaranteed to embed the config hash — and asks
    `is_complete`, so it is correct for any registry, not just directory-backed ones.
    Age is best-effort; an object-store registry reports the skip without a timestamp
    rather than failing. `is_complete` raising is **fatal**, because unknown skip state
    is not the same as no skips.
  - **`RunFnImportable`**: the registry path's analogue of `PayloadClosed`. A stale
    `'module:function'` reference fails identically on every leg, and there is no reason
    to discover that once per leg. Distinct from `ImportReady`, which is a fleet
    *readiness predicate* run over SSH against a rented box.
  - **`registry_gauntlet`**: the companion to `standard_gauntlet`, which cannot be
    reused — it is built around a provider, a host spec and an SSH key, none of which a
    local or in-cluster campaign has.
- **`RemoteEnvPinned` now documents what it cannot reach** (`gauntlet.py`): the RESOLVED
  state. It proves the executor is configured to *ship* a variable, not that the variable
  had the effect it was set for. Both halves fail silently and identically. An engine
  whose correctness depends on the effect should assert the effect next to its `RunFn` —
  set-but-wrong and right-by-accident both need to fail, and only the second is invisible
  today. Not added as a check here: run-farm cannot know what backend an engine needs, and
  a check that defaults to requiring nothing could not fail.

Found while farming a Morphospace calibration sweep, where the engine's own arm is
reproducible only on CPU: run it on the GPU and every leg returns entirely plausible
numbers whose reproducibility claim is void. That check belongs to the engine, but the
two above did not, and neither existed.

## [0.3.0] — The worker's environment, and a channel death that is not a work failure

The version had read `0.2.0` — the same string PyPI serves — while the branch sat ten
commits past the tag, so nothing distinguished a checkout from the published wheel. The
behaviour changes below are exactly what a downstream would silently have been missing,
with no wrong number anywhere to show for it. Confirmed on a second machine before this
release: an editable install there reported **`0.1.1`**, two releases stale, and had been
answering every `pip list` through a full GPU campaign.

A minor bump rather than a patch: every item below is in `Added`, and this project's rule
is *pre-1.0, minor = features*.

### Added
- **The worker's environment** (`provider_exec.py`):
  `ProviderExecutor(remote_env={...})` prefixes `env K=V …`, shell-quoted. It exists
  because the worker arrives over a separate **non-interactive** `ssh host cmd`, which
  sources no profile and inherits nothing the provider's onstart exported — so there was
  no way to set a variable the worker must see **at process start**. `XLA_FLAGS` is the
  motivating case: JAX reads it when the backend initialises, so setting it from inside a
  RunFn is already too late, and XLA GPU autotuning picks kernels per process. Measured on
  a Morphospace lineage at campaign scale, a cross-process resume diverged from the
  uninterrupted run in **1 of 3 attempts, 14587/20000 entries**;
  `--xla_gpu_autotune_level=0` makes it 5/5 identical. Applied to the readiness probe as
  well as the worker, so a variable that breaks `import` fails at the probe rather than
  once per leg at rental prices.
- **`RemoteEnvPinned`** (`gauntlet.py`): fails the gauntlet before anything is rented when
  the executor is not set to ship the variables a campaign needs. Worth a guard rather
  than a convention precisely because forgetting it is **silent** — every leg still runs
  and produces plausible numbers, and only the reproducibility claim is void. Checks
  set-but-WRONG as well as missing; not added to `standard_gauntlet`, because a check that
  defaults to requiring nothing could not fail.
- **Reattach in `ProviderExecutor` too** (`provider_exec.py`): the structured-`RunFn` path
  now makes the same transport-vs-work distinction `fleet.py` does.
  `ProviderExecutor(reattach_attempts=, reattach_backoff_s=)`, off by default. The gap was
  narrower than it looked — `run` **already** failed a config over when the provider
  *confirmed* the host died; what was uncovered was the opposite case, `rc 255` over a host
  still **alive**, recorded as a config error and lost with its checkpoint on a healthy
  box. Unlike a `FleetLeg`, the worker CLI owns no *launch-if-not-already-running*
  contract (it has no pidfile), so the executor supplies it: `_worker_gone` polls for a
  surviving worker and declines to start a second one, conservative on an ambiguous probe.
  Safe to re-invoke because `driver.execute_config` skips-if-complete else resumes.

### Fixed
- **`run-farm-reap` names the provider it scanned.** `--provider` defaults to `vast`, so
  running it bare after a RunPod campaign printed a confident *"no live instances —
  nothing to reap"* while a pod billed — observed 2026-08-04, with the pod `RUNNING` and
  visible through its own API at that moment. The unscoped wording *"ALL live instances"*
  compounded it by reading as all-instances-everywhere. Both lines now name the provider,
  and the empty case points at the one that was **not** checked. This is the reap tool; a
  reassuring message from it is the one that must not be wrong.

### Added (from the prior working set)
- **Reattach on a dead ssh channel** (`fleet.py`): `FleetLeg.reattachable` (opt-in,
  default off) plus `FleetExecutor(reattach_attempts=, reattach_backoff_s=)`. `rc 255`
  is ssh's own transport code and never came from the payload, so over a host the
  provider confirms alive it cannot mean the work failed. It was previously filed as a
  terminal `RUN_FAIL`, which tore the box down under it — measured cost, a 5500-step
  N=320 relaxation. All attempts share **one** `run_timeout` deadline, with the backoff
  clamped to it, so retries cannot multiply the billing window.
- **A budget cap that halts the campaign** (`protocols.py`, `fleet.py`):
  `BudgetExceeded` moved beside the other failover signals and re-raised out of `run()`
  instead of being buried by the catch-all as one `ERROR` per leg. Unstarted futures are
  cancelled — propagation alone still let every queued leg call `rent()`, and a budget
  halt that still rents is not a halt.
- **`skip=` on the gauntlet** (`gauntlet.py`): `run_gauntlet` / `require_gauntlet` accept
  it, and a skipped check is REPORTED as skipped rather than vanishing from the report.
  Two checks had been telling callers to pass a parameter that did not exist.
- `CITATION.cff`: version DOIs for **v0.1.1** (`10.5281/zenodo.21420305`) and **v0.2.0**
  (`10.5281/zenodo.21726314`). Both were minted and simply never recorded — v0.1.1's
  existence answers the open question of whether that tag ever got a GitHub Release: it
  did. Queried from the Zenodo API against the concept record rather than guessed from
  the numbering, and each one confirmed to resolve. The README badge stays on the
  concept DOI, which points at whatever the latest version is.

## [0.2.0] — Checks that can fail

The wrapper around the campaign mechanism, built from the three things the 2026-07 B2
campaign paid for. Four new public modules, additive only. All of them obey one rule:
**every check must be able to fail — if it cannot, it is not a check.**

### Added
- **Payload closure validator** (`payload.py`): `validate_flat`, `require_flat`,
  `PayloadSpec`, `PayloadError`. A `ship` tuple is copied FLAT into the worker cwd, so
  a repo laid out over several directories imports fine at home and dies on the box —
  and you pay a rental to find out. `validate_flat` stages the payload into a temp dir
  and runs it there, locally, for free. Five verdicts, each a reconstruction of a real
  failure: `missing_file`; `name_collision` (two dirs shipping one basename, and
  flattening keeps one); `import`, **including a module that resolves OUTSIDE the flat
  dir** — satisfied by the ambient environment, which is what made a fresh-clone test
  pass while testing nothing; `startup`, load-bearing because the original bug read a
  sibling inside `engine_sha()`, so `import` alone SUCCEEDED and the rental still died
  at exit=1; and `mismatch`, a payload that runs but computes something different from
  home. `env` exists because the launch command exports `ENGINE_COMMIT`, and a
  validator that cannot set it validates a different startup than the one that runs.
- **Launch gauntlet** (`gauntlet.py`): `run_gauntlet`, `require_gauntlet`,
  `standard_gauntlet`, `CheckResult`, `GauntletError`, and the checks —
  `SshKeyPresent`, `SshKeyRegistered`, `SshHandshake`, `PayloadClosed`,
  `OffersAvailable`, `OutDirWritable`, `ResumeMarkersIntended`, `ProviderCapable`.
  Everything that can fail before money is spent. `SshKeyPresent` is the 7-rental,
  72-minute failure: `FleetExecutor` defaults to `key_path=~/.ssh/vastai`, the file did
  not exist, every host refused the connection, and each refusal looked like a
  spot-pool failure — a Vast API call had stood in for an SSH test, and
  `list_instances()` never touches SSH. So `CheckResult` carries `proves`: what a green
  result establishes and what it does NOT. `SshKeyRegistered` FAILS when a provider
  exposes no `registered_ssh_keys()` rather than passing, because "I could not look"
  must never render as "it is fine". `ResumeMarkersIntended` reports every leg that
  will pre-skip, with marker age: a leg once reported SKIP because a DIFFERENT
  provider's marker satisfied `done_when`.
- **Provider diagnostics** (`diagnostics.py`): `Diagnostics`, `Gap`, `capabilities`,
  `collect`, `explain_failure`. Vast has `logs()`; RunPod does not — and a monitor that
  called `RunPodProvider.logs()` with stderr suppressed reported nothing, which looked
  like a healthy quiet host. `gaps` keeps "the host produced no logs" separate from
  "this provider has no `logs()` to ask", `alive` is three-valued so a failed call
  never becomes "not alive", and `collect()` never raises: diagnostics run on the error
  path, and one that throws destroys the report you were writing.
- **Arrival contract** (`arrival.py`): `verify_file` OPENS artifacts instead of
  trusting size and magic bytes. A truncated 86 MB `field.npz` had a plausible size and
  correct PK magic, and failed only on open. Zip-family members (`.npz`, `.zip`,
  `.whl`) are CRC-verified via `testzip()`, not merely opened: a flipped payload byte
  leaves the central directory intact, so a bare `ZipFile(p)` SUCCEEDS on a corrupt
  file. `.gz` is fully decompressed (its CRC32 is in the trailer) and `.tar`/`.tar.gz`
  members are read through, truncation being the failure that actually occurs.
- **`verify_report` / `ArrivalReport`**: the same walk, with what could NOT be checked
  counted rather than folded into the pass. Opaque formats (`.bin`, `.pt`, bare `.npy`
  without numpy) carry no internal checksum, so they are reported `unverifiable` — a
  report reading "all clear" over a directory of opaque blobs is the same overclaim in
  a different costume. Partial-write residue (`.tmp`, `.part`, `.partial`, …) and an
  empty fetch are surfaced as warnings, not failures: `done_when` is the caller's
  declaration of completeness, and overriding it here would redefine a caller's
  contract in the name of integrity.
- **`FleetExecutor` gates on arrival**: a new `BAD_ARTIFACTS` leg status, and
  `validate_artifacts=True` to opt out. A leg whose run succeeded and whose fetch
  landed used to be `OK` on the strength of a marker file existing — nothing opened the
  payload, so a transport fault arrived wearing the costume of a result. Distinct from
  `RUN_FAIL` because the remedy differs: re-fetch or re-run, versus fix the job.
  Validation runs only after the FINAL fetch; a `resumable` leg's periodic pull is
  *expected* to catch files the remote is still writing, so checking there would
  generate false alarms.
- 85 tests, every passing case paired with the failure it must catch (suite 242 -> 330),
  and pytest `pythonpath = ["src"]` — without it an installed `run_farm` shadows this
  checkout and a new module reads as missing.

### Fixed
- **`fleet._fetch` could publish a completion marker over an incomplete leg.** `scp -r`
  lands files in arbitrary order and the `done_when` marker is one of them, so a fetch
  that wrote the marker and then died left a leg dir `_complete()` reads as done —
  silently pre-skipping an incomplete leg on every future relaunch. `_fetch` now stages
  and publishes the marker LAST: a completion marker must be the last byte written or
  it is not one. Verified by reverting `_fetch` to the un-staged form, which fails all
  three new fleet tests and reports plain "LEG L1: OK" for a truncated `field.npz`.
- The three marker-last publication tests asserted a property of `sorted()` rather than
  of `publish`: their payload files sorted before the marker, so the marker landed last
  with the ordering code deleted. Renamed so the ordering is the only thing that can
  produce the result — verified by deleting both ordering sites and watching all three
  fail.

### Changed
- `FleetExecutor._fetch` no longer verifies each pull. `publish(..., verify=False)`:
  publication never withholds a file, so verifying there only produced a log line —
  once per mid-run pull, over files the remote was still writing, and blind to
  anything an earlier pull had already banked. `_validate_artifacts` does it once,
  over the whole leg dir, after the final fetch.
- `_fetch_loop`'s docstring no longer claims "the box writes atomically, so an
  in-flight pull never grabs a half-written file". That is a claim about the CALLER's
  engine which this class cannot make — and one such engine did not, which is how the
  86 MB truncation happened.
- **`run_farm.preflight` -> `run_farm.gauntlet`** (`PreflightError` -> `GauntletError`,
  `gauntlet()` -> `run_gauntlet()`, `require()` -> `require_gauntlet()`). Two things
  named preflight in one package is exactly the ambiguity that lets someone think a
  check ran when a different one did: `FarmCampaign(preflight=...)` is the injected
  DOMAIN envelope over a config dict, while the gauntlet is the LOCAL LAUNCH
  ENVIRONMENT. `farm.py`'s `preflight=` hook and `launch_gate` are untouched, and the
  rename happened before the tag, so no released name changes.
- Docs: `docs/RELEASING.md` corrected — `src/run_farm/__init__.py` carries no
  `__version__`. README status line now reads 0.2.x.

## [0.1.1] — Downstream-extraction fixes

Cut while extracting the campaign layer out of jax-solitons (its first real
external consumer). No API changes; a correctness fix and coverage the port missed.

### Fixed
- **`FileRunRegistry.load` rebuilt checkpoints with the wrong config type.** It used
  the default `SimpleRunConfig.from_json`, which raised `TypeError` on any engine
  whose `RunConfig` has its own fields (jax-solitons' `model`/`N`/`L`). It now
  rebuilds with the handle's own config type. Found by jax-solitons' bit-identical
  resume test across the package seam — exactly what a downstream integration test
  is for. Regression pinned in `tests/test_config.py`.

### Added
- Admission (E) and Provider-contract (F) tests migrated from jax-solitons'
  `test_campaign.py` — ProbeAdmission's probe-or-bail gates and FakeProvider
  teardown/failover — lifting `run_farm.reference` coverage the initial port left low.

### Changed
- Docs: "leak-proof" softened to **best-effort verified teardown** throughout, and a
  README **Cost and liability** section added. The teardown contract cannot be a
  guarantee — a SIGKILL or a crash in the create→track window can still orphan a
  billing host (reap is the backstop) — and claiming otherwise alongside the MIT
  no-liability clause was contradictory.


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
