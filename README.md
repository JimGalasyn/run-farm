# run-farm

[![CI](https://github.com/JimGalasyn/run-farm/actions/workflows/ci.yml/badge.svg)](https://github.com/JimGalasyn/run-farm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JimGalasyn/run-farm/branch/main/graph/badge.svg)](https://codecov.io/gh/JimGalasyn/run-farm)
[![CodeQL](https://github.com/JimGalasyn/run-farm/actions/workflows/codeql.yml/badge.svg)](https://github.com/JimGalasyn/run-farm/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/JimGalasyn/run-farm?include_prereleases)](https://github.com/JimGalasyn/run-farm/releases)
[![PyPI](https://img.shields.io/pypi/v/run-farm)](https://pypi.org/project/run-farm/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21419776.svg)](https://doi.org/10.5281/zenodo.21419776)
[![Python](https://img.shields.io/pypi/pyversions/run-farm)](https://pypi.org/project/run-farm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Status: alpha (0.4.x).** The API will change without notice until 1.0.
>
> **⚠ This tool spends real money.** It rents billable cloud GPUs on your accounts.
> You are solely responsible for all charges it incurs. See
> [Cost and liability](#cost-and-liability) before running a live campaign.

**Checkpointed, config-hashed campaign runs over a rented GPU spot fleet.** You bring
an engine (any `Callable[[RunConfig, RunContext], dict]`) and a config; run-farm gives
you a restart-exact run registry, streamed event records, probe-or-bail admission on
flaky marketplace hosts, and cloud brokers for Vast, RunPod, and Modal that make a
**best-effort** to tear down and verify every rented host.

## Why

Spot-fleet executors (SkyPilot, dstack) recover from preemption but assume reliable
hosts and own no provenance. A 2026-06 literature sweep found **nothing** covering the
full contract run-farm owns:

- **Restart-exact, config-hashed runs.** A result names the config hash that produced
  it; a preempted run resumes bit-identically from a full-state checkpoint, and
  re-submitting a finished run is a no-op. Idempotent skip makes spot preemption free.
- **Probe-or-bail admission (P9).** Hosts, networks, and devices lie -- the standing
  case study is a 0.996-reliability host with **zero** outbound bandwidth. run-farm
  measures a host before it runs work and bails on the ones that can't. Measured live:
  **58% of created instances never boot**, and probe-or-bail fails them over ~9× faster
  than waiting out a timeout -- so P9 is a cost feature, not just a correctness one.
- **Teardown-verifying cloud brokers (P10).** A leaked GPU bills by the second. Every
  `Provider.rent()` tears its host down on exit — normal, exception, or Ctrl-C — and
  independently re-checks that it is gone, raising loudly if it can't confirm. This is
  **best-effort, not a guarantee**: a hard kill (SIGKILL, power loss) or a crash in the
  window between *creating* an instance and *tracking* it can still orphan a billing
  host. Always run `run-farm-reap` after a campaign to catch strays, and set a
  [budget cap](#cost-and-liability). In live testing the normal, exception, and SIGTERM
  paths tore down cleanly; the create-window gap is real and is why reap exists.

## Install

```bash
pip install run-farm                 # base: jax + numpy only
pip install 'run-farm[modal]'        # + the Modal serverless executor
pip install 'run-farm[s3]'           # + the S3 object-store registry backend
```

The Vast broker is intentionally stdlib-only (the vastai SDK breaks against the live
API), so there is no `vast` extra.

## Quickstart

```python
from run_farm import (SimpleRunConfig, FileRunRegistry, JsonlEventSink,
                      LocalExecutor, ProbeAdmission, run_campaign)
from run_farm.testing import echo_run_fn      # a physics-free RunFn

configs = [SimpleRunConfig(name="demo", params={"i": i}) for i in range(4)]
run_campaign(configs, echo_run_fn,
             registry=FileRunRegistry("out"), sink=JsonlEventSink(),
             admission=ProbeAdmission(require_gpu=False), executor=LocalExecutor())
```

Swap `LocalExecutor` for a `ProviderExecutor` over `VastProvider`, a `FleetExecutor`,
or a `ModalExecutor` -- the RunFn and records are unchanged. When you do, size the host
by memory and not by name: **a `gpu_name` is not a memory spec** ("A100 SXM4" is sold as
both 40 GB and 80 GB, and cheapest-first ordering reliably returns the 40 GB card), so
set `HostSpec(min_gpu_ram_mb=...)` rather than discovering the mismatch after paying to
provision the wrong hardware. `run_farm.testing` ships
physics-free RunFns so you can smoke-test a real fleet end to end for pennies before
pointing an expensive engine at it.

## Modules

| Module | What |
|---|---|
| `protocols` | the six contracts (RunConfig, RunRegistry, EventSink, Admission, Executor, Provider) |
| `config` | `SimpleRunConfig` + restart-exact checkpoint / run-directory helpers |
| `driver` | the physics-blind `run_campaign` / `execute_config` |
| `reference` | local-machine `FileRunRegistry`, `JsonlEventSink`, `ProbeAdmission` |
| `vast`, `runpod` | reference `Provider` adapters (best-effort teardown + verify) |
| `provider_exec`, `fleet` | rent-a-box executors, with per-host failover |
| `modal_exec` | serverless executor (needs `[modal]`) |
| `store` | shared object-store registry/sink for cross-cloud campaigns (needs `[s3]`) |
| `ledger` | `RentalLedger` -- append-only rental receipts (spend + outcomes) |
| `budget` | `estimate()` + `CappedProvider` (an enforced dollar cap) |
| `sweep` | `legs()` -- expand (arm × replicate × grid) into a campaign |
| `reap` | destroy orphaned instances (scoped, refuses unsafe sweeps) |
| `testing` | physics-free RunFns for engine-less smoke tests |
| `payload` | does what you SHIP run where it LANDS? flat-layout validation, locally |
| `gauntlet` | the launch gauntlet -- everything that can fail before money is spent, on both the fleet and registry paths |
| `diagnostics` | uniform host diagnostics, with provider capability GAPS named |
| `arrival` | verify fetched artifacts; publish so a marker never precedes its payload |

## The launch gauntlet: fail locally, not on a rented box

Campaign failures are rarely physics. They are a key that was never registered, a
payload one file short, a stale marker that made a skipped leg read as a passing one.
Each is free to detect locally, and each has instead been detected by renting GPUs.

```python
from run_farm import PayloadSpec, standard_gauntlet, require_gauntlet

require_gauntlet(standard_gauntlet(
    provider=provider, host_spec=spec, out_dir="out/", legs=legs,
    key_path="~/.ssh/vastai",
    payload=PayloadSpec(
        files=SHIP, imports=("standard_box", "core_knot_id"),
        startup="import standard_box as sb; print(sb.engine_sha()[0])",
        env=(("ENGINE_COMMIT", commit),), expect=local_sha),
))                       # raises GauntletError listing EVERY blocker, before renting
```

`standard_gauntlet` serves the **fleet** path — rent hosts, ship a payload, run
`FleetLeg`s. `run_campaign` over a `RunRegistry` is the other first-class path, and it
has no provider, host spec or SSH key to check; `registry_gauntlet` is its companion:

```python
from run_farm import registry_gauntlet, require_gauntlet

require_gauntlet(registry_gauntlet(
    registry=registry, configs=configs, out_dir=out_dir,   # out_dir IS the registry base
    run_fn_ref="my_engine.runfns:my_run",
))
```

It answers the same question the fleet path asks about `done_when` markers — **which
legs are about to be skipped, and how old is the evidence** — against
`RunRegistry.is_complete` instead. `RegistryMarkersIntended` is read-only on purpose:
`run_campaign` defers registration to the worker that picks a config up, and a check
that pre-registered would both reintroduce that cost and create run directories for
work that never happens.

What it deliberately does *not* check is whether the compute backend is the one your
science needs — run-farm cannot know that, and a check defaulting to requiring nothing
could not fail. Assert it next to your `RunFn`, against the **resolved** state rather
than the variable meant to set it: set-but-wrong and right-by-accident both need to
fail, and only the second is invisible.

The governing rule, learned expensively: **every check must be able to fail. If it
cannot, it is not a check.** A Vast API call once stood in for an SSH test — the key
was fine, `~/.ssh/vastai` did not exist, and 7 rentals over 72 minutes each looked
like a bad host. So every `CheckResult` carries `proves`: what a green result
establishes, and what it does not. The gauntlet cannot prove SSH works (that needs a
host, and a host costs money); it says so, and `SshHandshake` covers the rest once you
hold one.

Named `gauntlet`, not `preflight`, on purpose: `FarmCampaign(preflight=...)` is a
different layer — the injected *domain envelope* on a config dict ("can this
configuration hold?"), where this is the local launch environment. Neither
substitutes for the other.

The same distinction one layer out: **"the marker arrived" is not "the job worked."**
`LegResult.ok` is a statement about *transport*. Three real legs reached OK while
measuring nothing — a marker reading `exit=1` on a 51-minute rental, an OOM that left a
one-sample manifest scoring as a pass, and a payload that NaN'd and exited 0 with a
perfect marker. The last is not exit-code-visible, so the check has to be about the
**result**, and what a result looks like is payload-specific: set `FleetLeg.verdict` to
a callable over the output directory. A verdict that *raises* is itself a verdict — the
checker could not read the output, and calling that OK is the bug it exists to stop.

Related, same principle: `diagnostics.collect` distinguishes "the host produced no
logs" from "this provider has no `logs()` to ask" — a monitor once called a method
that does not exist with stderr suppressed, and silence read as health. And
`arrival.verify_file` **opens** artifacts, because a truncated 86 MB `.npz` had a
plausible size and correct magic bytes and failed only on open — and for zip-family
files it CRC-checks the members, since a flipped payload byte leaves the central
directory intact and a bare `ZipFile(p)` succeeds. `arrival.verify_report` counts what
it could NOT check (`.bin`, `.pt`, anything with no internal checksum) as
`unverifiable` rather than folding it into the pass. `FleetExecutor` gates on this:
a leg whose payload did not survive the trip reports `BAD_ARTIFACTS`, not `OK`.

## Development

```bash
pip install -e '.[test]'
pytest -q -n auto --cov=run_farm
```

## Cost and liability

run-farm rents **real, billable** cloud instances on marketplaces like Vast.ai,
RunPod, and Modal using **your** credentials. Running a campaign spends your money.
**You are solely responsible for every charge it incurs, whatever the cause** —
including bugs, crashes, network failures, marketplace misbehavior, orphaned or
leaked instances, misconfiguration, or a campaign that simply costs more than you
expected.

The safety mechanisms are **best-effort, not guarantees**:

- **Teardown** fires on normal, exception, Ctrl-C and `SIGHUP` exits and re-verifies
  the host is gone. `SIGHUP` is the one that matters in practice: a campaign run from a
  terminal, an ssh session or an agent shell takes it when that session ends, and its
  default disposition is *terminate* — so before 0.4.0 the interpreter died outright and
  ran no teardown at all. But a `SIGKILL`, a power loss, or a crash in the create→track
  window can still leave a billing host alive. **Run `run-farm-reap` after every
  campaign** to find and destroy strays, and pass `--ledger` so rows left open by a dead
  driver are closed too — an unclosed `rented` row counts as burning forever and will
  eventually refuse every rent against that ledger.
- **`CappedProvider`** refuses to *start* a rental once spend reaches the cap, but it
  is a pre-rent gate, not a mid-rental tripwire: a rental already running can still
  overshoot by its own runtime, and the cap depends on the ledger being accurate. A cap
  set *below* the campaign's own worst case is worse than either finishing or refusing
  to start — it pays for whatever ran — so `CapClearsWorstCase` checks for that at
  launch, non-fatally, since a tight cap you intend to babysit is legitimate.
- **`estimate()`** is an estimate. Real cost depends on host failure rates,
  marketplace pricing, and how long your work actually runs — all of which vary.

Recommended before any spend: set a `CappedProvider` cap, keep a `RentalLedger`,
watch live burn with `run-farm-status`, and reap when done. None of this removes your
responsibility for the bill.

This software is provided under the MIT License **"as is", without warranty of any
kind, and with no liability** to the authors for any damages — including money lost
on live campaigns. See [`LICENSE`](LICENSE) for the controlling terms; this section
is a plain-language summary, not a modification of them.

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE). Note the warranty and liability disclaimers, which
are load-bearing for a tool that spends money: see [Cost and liability](#cost-and-liability).
