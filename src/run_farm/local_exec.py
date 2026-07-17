"""InProcessExecutor: run a campaign slice on THIS machine's GPU, in-process.

The local analog of `ModalExecutor`/`ProviderExecutor` for `run_multi` -- same
`run(configs) -> list[dict]` shape -- so the local GPU is just another
participant in a multi-provider partition, its results merged by
content-addressed identity (`config_hash`) exactly like any cloud's. No rental,
no teardown; it runs the shared `run_one` unit against a local file registry, so
the semantics are identical to a remote worker -- the only difference is the
machine.

(Distinct from `reference.LocalExecutor`, which is the in-process *thunk*
executor for `run_campaign`; this one takes configs and returns records, the
shape the multi-provider runner consumes.)

    from run_farm.local_exec import InProcessExecutor
    from run_farm import split_configs, run_multi
    local = InProcessExecutor("run_farm.testing:echo_run_fn")
    run_multi(split_configs(configs, [local, modal, vast, runpod]))  # local + clouds
"""

from __future__ import annotations

from collections.abc import Iterable

from run_farm.remote import RunFnRef, run_one
from run_farm.protocols import RunConfig


class InProcessExecutor:
    """Run each config on the local machine via the shared `run_one` unit.

    `run_fn_ref` is the ``'module:function'`` RunFn (same injection the remote
    workers import); `work_dir` roots the local file registry/sink. Configs run
    sequentially on whatever device jax sees here (a local GPU if present).
    """

    name = "local"

    def __init__(self, run_fn_ref: RunFnRef, *,
                 work_dir: str = "campaign_out/local"):
        self.run_fn_ref = run_fn_ref
        self.work_dir = work_dir

    def run(self, configs: Iterable[RunConfig], *, admission=None) -> list[dict]:
        """Run the slice in-process; returns one result record per config.

        `admission` (the P9 probe-or-bail) is accepted for interface parity but
        is a no-op here -- there is no flaky rented host to probe; it is your own
        machine.
        """
        return [run_one(c.to_json(), self.run_fn_ref, self.work_dir)
                for c in configs]
