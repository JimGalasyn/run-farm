"""A runnable campaign that needs no engine and no GPU -- the 60-second smoke test.

Drives the full A/B/C contract locally over a physics-free RunFn: register each
config, checkpoint per step, stream events, finish. Swap `LocalExecutor` for a
`ProviderExecutor` over `VastProvider` (and `echo_run_fn` for your own
'module:function' RunFn) and the same campaign fans out over a rented spot fleet --
the RunFn and the records are unchanged.

    python examples/local_campaign.py [out_dir]
"""

import sys

from run_farm import (FileRunRegistry, JsonlEventSink, LocalExecutor,
                      ProbeAdmission, SimpleRunConfig, run_campaign)
from run_farm.testing import counting_run_fn


def main(out: str = "campaign_out") -> None:
    registry = FileRunRegistry(out)
    sink = JsonlEventSink()
    admission = ProbeAdmission(require_gpu=False)   # set True (default) on a fleet
    executor = LocalExecutor()                      # -> ProviderExecutor at scale

    configs = [SimpleRunConfig(name="demo", params={"steps": 4, "i": i})
               for i in range(3)]

    run_campaign(configs, counting_run_fn, registry=registry, sink=sink,
                 admission=admission, executor=executor)

    print(f"wrote {len(configs)} config-hashed run dirs + MANIFEST under {out}/")
    for c in configs:
        print(f"  {c.run_name()}")
    # A re-run is the idempotent skip -- every run is already complete.
    run_campaign(configs, counting_run_fn, registry=registry, sink=sink,
                 admission=admission, executor=executor)
    print("re-ran: all skipped (idempotent), as a preempted campaign would resume")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "campaign_out")
