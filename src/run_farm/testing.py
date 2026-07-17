"""Physics-free `RunFn`s: exercise the whole campaign contract with no engine.

Shipped IN THE PACKAGE, not in tests/, for a load-bearing reason: a `RunFn`
crosses the wire as an importable ``'module:function'`` ref, so a dummy has to be
importable ON THE BOX too. `tests/` isn't on a rented host; `run_farm.testing` is
(it installs with the package). That makes these the right way to smoke-test a real
fleet end to end -- rent, ship, run, sync, tear down -- for pennies, before you
point an expensive engine at a config that might be wrong.

They also stand in for the engine in run-farm's own tests, which is why they live
here rather than being duplicated per test module.

Each takes `(config, ctx)` and returns a small JSON-safe record, exactly as the
`RunFn` contract requires. `numpy` is used only to make checkpoint state a real
array (so the .npz round-trip is exercised); no jax, no physics.
"""

from __future__ import annotations

import numpy as np


def echo_run_fn(config, ctx) -> dict:
    """The minimal honest RunFn: emit one event, checkpoint once, return a record.

    Resume-aware: on a fresh run it checkpoints a single scalar state at step 1; on
    a resumed run it reports the step it resumed from. Enough to prove the A/B/C
    wiring (register -> checkpoint -> finish, and resume feeding `ctx.resume`) with
    zero compute.
    """
    resumed_from = ctx.resume_step
    ctx.emit({"kind": "echo", "params": config.params,
              "resumed_from": resumed_from})
    if ctx.resume is None:
        ctx.checkpoint({"x": np.asarray([0.0], dtype=np.float64)}, 1)
    return {"ok": True, "resumed_from": resumed_from,
            "run": config.run_name()}


def counting_run_fn(config, ctx) -> dict:
    """Checkpoint EVERY step up to `params['steps']`, resuming from `ctx.resume`.

    Proves contract B (full-state restart) without physics: the state is a single
    running counter, checkpointed each step, so a resume must continue the count
    rather than restart it. `tests` assert the final count equals `steps`
    regardless of where a simulated preemption cut the first attempt.
    """
    steps = int(config.params.get("steps", 3))
    if ctx.resume is not None:
        count = float(ctx.resume["count"][0])
        start = (ctx.resume_step or 0) + 1
    else:
        count, start = 0.0, 1
    for step in range(start, steps + 1):
        count += 1.0
        ctx.checkpoint({"count": np.asarray([count], dtype=np.float64)}, step)
        ctx.emit({"kind": "count", "step": step, "count": count})
    return {"final_count": count, "steps": steps, "run": config.run_name()}


def failing_run_fn(config, ctx) -> dict:
    """Always raises -- models a preemption/crash mid-run.

    Used to prove the executor's failure path: the sink is still flushed, the run
    is NOT marked complete, and re-submitting it (idempotent skip won't fire) runs
    it again. `params['emit_before_fail']` optionally streams one event first, to
    check that partial event records survive a crash.
    """
    if config.params.get("emit_before_fail"):
        ctx.emit({"kind": "pre-crash", "params": config.params})
    raise RuntimeError(f"failing_run_fn: simulated crash for {config.run_name()}")
