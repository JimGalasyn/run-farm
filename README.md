# run-farm

> **Status: alpha (0.1.x).** The API will change without notice until 1.0.

**Checkpointed, config-hashed campaign runs over a rented GPU spot fleet.** You bring
an engine (any `Callable[[RunConfig, RunContext], dict]`) and a config; run-farm gives
you a restart-exact run registry, streamed event records, probe-or-bail admission on
flaky marketplace hosts, and leak-proof, teardown-verified cloud brokers for Vast,
RunPod, and Modal.

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
- **Leak-proof cloud brokers (P10).** A leaked GPU bills by the second. Every
  `Provider.rent()` destroys its host on *every* exit and independently verifies it is
  gone, raising on a leak. Verified live: **0 leaks across 34 rentals**, including
  through SIGTERM.

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
| `vast`, `runpod` | reference `Provider` adapters (leak-proof, teardown-verified) |
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

## Citing

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
