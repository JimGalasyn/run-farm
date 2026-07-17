"""Campaign axes: expand (arms x replicate seeds x a param grid) into legs.

The unit of a replicate campaign is not one run -- it is a *cross-product*. The
canonical experiment (Gould's replaying-the-tape: many lineages from a common
ancestor under identical selection, compared across arms) is meaningless without
two axes being first-class:

  arm         a named condition -- control vs treatment, or a built-in calibration
              gate that travels with every campaign (e.g. a known-answer reference
              run). One of your arms IS the gate; it can't be a params afterthought.
  replicate   the same starting point, a different RNG stream. Divergence ACROSS
              replicates under a FIXED arm is the signal; you cannot measure it
              without replicate being an axis you can group records by.

`legs()` is the expansion; it does NOT run anything and does NOT aggregate.
Aggregation (multi-seed means, A/B Wilcoxon, Wilson CIs) is the consumer's job --
that's scipy over the result records, not orchestration. run-farm's job is only to
make (arm, replicate) present and groupable on every leg it emits.

⚠ `seed_fn` is CALLER-SUPPLIED and never derived here. A consumer's seed-draw order
can be load-bearing for reproducibility of already-published results, and different
consumers have different collision-free schemes. Re-deriving seeds inside run-farm
would silently invalidate results. The farm carries the seed; it does not choose it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from itertools import product
from typing import Any


def legs(arms: Sequence[str],
         replicates: Sequence[int],
         grid: Iterable[Mapping[str, Any]] | None = None,
         *,
         seed_fn: Callable[[str, int, Mapping[str, Any]], int],
         rid_fn: Callable[[str, int, Mapping[str, Any]], str] | None = None,
         ) -> Iterator[dict]:
    """Expand the cross-product of `arms` x `replicates` x `grid` into leg dicts.

    Each yielded leg carries its axes explicitly so a downstream aggregation can
    ``groupby`` them straight off the result records::

        {"rid": <unique id>, "arm": <arm>, "replicate": <rep>,
         "seed": seed_fn(arm, rep, cell), "cfg": {<the grid cell>}}

    `grid` is an iterable of parameter-cell mappings (one dict per point in a
    parameter sweep); omit it (or pass a single empty mapping) for a pure
    arm x replicate campaign. `seed_fn(arm, replicate, cell) -> int` is required --
    see the module note on why run-farm won't invent it. `rid_fn` names each leg;
    the default is a stable, human-legible composite.

    Order is deterministic: arms outermost, then replicates, then grid -- so two
    runs of the same inputs enumerate identically (a re-launch resumes cleanly).
    """
    cells = list(grid) if grid is not None else [{}]
    if rid_fn is None:
        def rid_fn(arm, rep, cell):
            tail = "_".join(f"{k}{cell[k]}" for k in sorted(cell))
            return f"{arm}_r{rep}" + (f"_{tail}" if tail else "")

    for arm, rep, cell in product(arms, replicates, cells):
        cell = dict(cell)                    # defensive copy: caller may mutate/reuse
        yield {
            "rid": rid_fn(arm, rep, cell),
            "arm": arm,
            "replicate": rep,
            "seed": seed_fn(arm, rep, cell),
            "cfg": cell,
        }
