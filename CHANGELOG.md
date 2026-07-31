# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this project follows
[Semantic Versioning](https://semver.org/) (pre-1.0: minor = features).

## [Unreleased]

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

### Fixed
- **`fleet._fetch` could publish a completion marker over an incomplete leg.** `scp -r`
  lands files in arbitrary order and the `done_when` marker is one of them, so a fetch
  that wrote the marker and then died left a leg dir `_complete()` reads as done —
  silently pre-skipping an incomplete leg on every future relaunch. `_fetch` now stages
  and publishes the marker LAST: a completion marker must be the last byte written or
  it is not one. Verified by reverting `_fetch` to the un-staged form, which fails all
  three new fleet tests and reports plain "LEG L1: OK" for a truncated `field.npz`.

### Changed
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
