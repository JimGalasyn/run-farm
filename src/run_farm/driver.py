"""The campaign driver: wire the four protocols around an injected `RunFn`.

`run_campaign` is the whole orchestration loop, and it is physics-blind -- the
only soliton-specific value is `run_fn`. Each config becomes a task thunk that
the `Executor` runs on the fleet; the thunk skips finished runs (idempotent
preemption recovery), resumes from the latest full-state checkpoint, and hands
the physics a `RunContext` wired to the registry and sink.
"""

from __future__ import annotations

from collections.abc import Iterable

from run_farm.protocols import (
    Admission,
    EventSink,
    RunConfig,
    RunContext,
    RunFn,
    RunRegistry,
)
from run_farm.protocols import Executor  # noqa: F401  (re-exported)


def execute_config(config: RunConfig, run_fn: RunFn, *,
                   registry: RunRegistry, sink: EventSink) -> dict | None:
    """Run ONE config end to end against (registry, sink) and return its result.

    register (A) -> skip if already complete, returning None (idempotent
    preemption recovery, D) -> resume from full state or start fresh (B) -> run
    the physics with a wired RunContext (C) -> finish. The sink is flushed even
    if the physics raises (a preemption modeled as an exception), so no DONE is
    written but the events persist and the run resumes.

    This is the campaign's unit of work, factored out of `run_campaign` so a
    REMOTE worker (Modal function, rented SSH box) can invoke the exact same
    semantics with a locally-built, shared-storage-backed registry -- the only
    difference being which machine it runs on. See `campaign.remote`.
    """
    handle = registry.register(config)
    if registry.is_complete(handle):
        return None                               # preemption no-op (P4/D)
    resume = registry.load(handle)
    ctx = RunContext(
        resume=None if resume is None else resume[0],
        resume_step=None if resume is None else resume[1],
        checkpoint=lambda state, step: registry.save(handle, state, step),
        emit=lambda record: sink.emit(handle, record),
        trigger=lambda state, reason: sink.trigger(handle, state, reason),
    )
    try:
        result = run_fn(config, ctx)
        registry.finish(handle, result)
        return result
    finally:
        sink.close(handle)


def run_campaign(
    configs: Iterable[RunConfig],
    run_fn: RunFn,
    *,
    registry: RunRegistry,
    sink: EventSink,
    admission: Admission,
    executor: Executor,
) -> None:
    """Run every config through `run_fn` over `executor`, with restart + records.

    Per run, ON THE WORKER that picks it up: register (A) -> skip if complete
    (D recovery) -> resume from full state or start fresh (B) -> run physics
    with a wired RunContext (C) -> finish. Admission (E) is enforced by the
    executor on each worker before any task runs.

    Registration is lazy: the run dir + manifest line are written by the worker
    that runs the config, not pre-flighted on the submitting node. At 10^4-10^6
    scale an eager `[register(c) for c in configs]` would serialize that many
    mkdir + manifest appends on one node before any work starts. The task thunks
    stream too -- a generator is handed to the Executor (`Executor.run` takes an
    Iterable), so neither the configs nor the thunks are materialized here; how
    far the work queue itself streams is then the Executor's choice.
    """
    def task_for(config):
        # Each task runs the unit of work on whichever worker picks it up; the
        # register/skip/resume/finish logic lives in execute_config (shared with
        # the remote workers in campaign.remote). The thunk discards
        # execute_config's return value to honor Executor.run's Callable[[], None]
        # contract -- executors must not depend on a task's return.
        def task() -> None:
            execute_config(config, run_fn, registry=registry, sink=sink)
        return task

    executor.run((task_for(c) for c in configs), admission)   # lazy generator
