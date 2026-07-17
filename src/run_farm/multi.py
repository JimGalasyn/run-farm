"""Drive ONE campaign across several providers/executors at once.

Partition a config list across remote executors (a `ModalExecutor`, a
`ProviderExecutor` over Vast, another over RunPod, ...), run them concurrently,
and merge the result records into one harvest. This is *sensible* precisely
because run identity is **content-addressed** (`RunConfig.config_hash`): a result
record names the config that produced it regardless of which cloud ran it, so
the merge cannot collide and is provider-independent.

This is the partition-and-merge tier, and it works today. It assumes a
**disjoint** partition (each config to one executor): `is_complete`/resume are
per-executor here, so two executors handed the same config would both run it.
Global dedup, cross-provider resume, and a single artifact store need a shared
`RunRegistry` backend (an object-store impl of A/B; see CAMPAIGN.md) -- a second
backend, not a new seam. Until then, partition disjointly and consolidate the
per-executor artifact dirs afterward (collision-free, since they're hashed).

    from run_farm.multi import split_configs, run_multi
    report = run_multi(split_configs(configs, [modal_ex, vast_ex, runpod_ex]))
    print(report.summary())
    winners = {name: r for name, r in report.results.items() if r["result"]}
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
from collections.abc import Callable, Iterable, Iterator, Sequence

from run_farm.protocols import RunConfig

# A "remote executor" here is anything with a `.name` and a
# `run(configs) -> list[dict]` (ModalExecutor, ProviderExecutor). Kept duck-typed
# rather than a Protocol import so this module stays dependency-light.
Assignment = tuple[object, list[RunConfig]]


@dataclasses.dataclass
class ProviderRun:
    """One executor's slice of the campaign and how it went."""

    provider: str
    n_assigned: int
    records: list[dict] = dataclasses.field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclasses.dataclass
class CampaignReport:
    """The merged harvest plus per-provider accounting.

    `results` maps `run_name` (config hash) -> the result record, annotated with
    the `provider` that produced it. `duplicates` flags any run_name produced by
    more than one provider -- a sign the partition wasn't disjoint.
    """

    results: dict[str, dict]
    by_provider: list[ProviderRun]
    duplicates: dict[str, list[str]] = dataclasses.field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.by_provider)

    def summary(self) -> str:
        lines = []
        for p in self.by_provider:
            tag = "ok" if p.ok else f"ERROR: {p.error}"
            done = sum(1 for r in p.records if not r.get("skipped"))
            lines.append(f"  {p.provider}: {p.n_assigned} assigned, "
                         f"{done} ran, {len(p.records) - done} skipped [{tag}]")
        head = (f"campaign: {len(self.results)} runs across "
                f"{len(self.by_provider)} provider(s); "
                f"{'OK' if self.ok else 'PARTIAL — see errors'}")
        if self.duplicates:
            head += f"; {len(self.duplicates)} DUPLICATE run(s) (non-disjoint split)"
        return "\n".join([head, *lines])


def split_configs(configs: Iterable[RunConfig], executors: Sequence[object],
                  weights: Sequence[float] | None = None) -> list[Assignment]:
    """Partition `configs` across `executors`, disjointly, into assignments.

    Round-robin by default; pass `weights` (one per executor) to bias the split
    toward faster/cheaper clouds. Deterministic given the inputs.
    """
    configs = list(configs)
    if not executors:
        raise ValueError("need at least one executor")
    buckets: list[list[RunConfig]] = [[] for _ in executors]
    if weights is None:
        for i, c in enumerate(configs):
            buckets[i % len(executors)].append(c)
    else:
        if (len(weights) != len(executors) or sum(weights) <= 0
                or any(w < 0 for w in weights)):
            raise ValueError(
                "weights must be one non-negative value per executor with "
                "positive sum (a negative weight gives nonsense bucket counts)")
        total = sum(weights)
        # Largest-remainder apportionment so the counts sum to len(configs).
        exact = [w / total * len(configs) for w in weights]
        counts = [int(x) for x in exact]
        for idx in sorted(range(len(executors)), key=lambda i: exact[i] - counts[i],
                          reverse=True)[:len(configs) - sum(counts)]:
            counts[idx] += 1
        pos = 0
        for b, n in enumerate(counts):
            buckets[b] = configs[pos:pos + n]
            pos += n
    return list(zip(executors, buckets))


def _name(ex) -> str:
    return getattr(ex, "name", type(ex).__name__)


def stream_multi(assignments: Iterable[Assignment], *,
                 max_workers: int | None = None) -> "Iterator[ProviderRun]":
    """Yield each executor's `ProviderRun` **as it completes** (completion order).

    The streaming primitive: a fast cloud's slice is handed back the instant it
    finishes, not gated on the slowest executor. Each executor runs in its own
    thread (its `.run` blocks on its cloud and does its own internal fan-out); a
    provider that raises yields a `ProviderRun` carrying the error rather than
    propagating -- one cloud going down never stops the others' results landing.
    Records are annotated with the producing `provider`.
    """
    assignments = [(ex, list(cfgs)) for ex, cfgs in assignments if cfgs]
    if not assignments:
        return
    with cf.ThreadPoolExecutor(max_workers=max_workers or len(assignments)) as pool:
        futs = {pool.submit(ex.run, cfgs): (ex, cfgs) for ex, cfgs in assignments}
        for fut in cf.as_completed(futs):
            ex, cfgs = futs[fut]
            name = _name(ex)
            try:
                records = fut.result()
            except Exception as e:  # noqa: BLE001 -- isolate one cloud's failure
                yield ProviderRun(name, len(cfgs), error=f"{type(e).__name__}: {e}")
                continue
            yield ProviderRun(name, len(cfgs),
                              records=[{**r, "provider": name} for r in records])


def run_multi(assignments: Iterable[Assignment], *,
              max_workers: int | None = None,
              on_result: "Callable[[ProviderRun], None] | None" = None,
              ) -> CampaignReport:
    """Run each executor's slice concurrently and merge into one harvest.

    Drains `stream_multi`, so providers' slices are merged in completion order;
    pass `on_result` to observe each `ProviderRun` live as it lands (progress
    printout, incremental write-out) without giving up the final merged report.
    A provider that raises is isolated -- recorded, others still harvested.
    """
    results: dict[str, dict] = {}
    duplicates: dict[str, list[str]] = {}
    reports: list[ProviderRun] = []
    for pr in stream_multi(assignments, max_workers=max_workers):
        if on_result is not None:
            on_result(pr)
        reports.append(pr)
        for r in pr.records:
            run = r.get("run")
            if run in results:                          # non-disjoint partition
                duplicates.setdefault(run, [results[run]["provider"]]).append(
                    pr.provider)
            results[run] = r
    return CampaignReport(results, reports, duplicates)
