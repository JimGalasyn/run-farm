# run-farm

[![CI](https://github.com/JimGalasyn/run-farm/actions/workflows/ci.yml/badge.svg)](https://github.com/JimGalasyn/run-farm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/JimGalasyn/run-farm/branch/main/graph/badge.svg)](https://codecov.io/gh/JimGalasyn/run-farm)
[![CodeQL](https://github.com/JimGalasyn/run-farm/actions/workflows/codeql.yml/badge.svg)](https://github.com/JimGalasyn/run-farm/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/JimGalasyn/run-farm?include_prereleases)](https://github.com/JimGalasyn/run-farm/releases)
[![PyPI](https://img.shields.io/pypi/v/run-farm)](https://pypi.org/project/run-farm/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21419776.svg)](https://doi.org/10.5281/zenodo.21419776)
[![Python](https://img.shields.io/pypi/pyversions/run-farm)](https://pypi.org/project/run-farm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Status: alpha (0.1.x).** The API will change without notice until 1.0.
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
or a `ModalExecutor` -- the RunFn and records are unchanged. `run_farm.testing` ships
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

- **Teardown** fires on normal, exception, and Ctrl-C exits and re-verifies the host
  is gone — but a `SIGKILL`, a power loss, or a crash in the create→track window can
  still leave a billing host alive. **Run `run-farm-reap` after every campaign** to
  find and destroy strays.
- **`CappedProvider`** refuses to *start* a rental once spend reaches the cap, but it
  is a pre-rent gate, not a mid-rental tripwire: a rental already running can still
  overshoot by its own runtime, and the cap depends on the ledger being accurate.
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
