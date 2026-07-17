"""RentalLedger: append-only JSONL receipts of host rentals (P9 -- write what you
measured).

Lifted out of `vast.py` because it never was Vast-specific: `RunPodProvider`
already takes one, and the `Provider` contract only asks for an object with a
`record(event, **fields)` method. It is the infrastructure analogue of the
campaign `EventSink` -- one JSON object per line for every `rented`, `running`,
and `destroyed` event, the last carrying outcome + billed seconds + est cost.

It is also the ONLY structure that knows about an instance during the window where
it can leak: a Provider logs `rented` (with the instance id) the moment the host
is created, then blocks in `wait_running` for minutes before the executor tracks
it in memory. A SIGTERM or crash in that window orphans a billing host that
in-memory tracking never saw -- so an orphan-sweep that consults the ledger, not
just live tracking, is the complete one. (Enforcement seam: `run_farm.budget`.)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# The default lives under the tool's own dotdir. Real campaigns pass an explicit
# per-campaign path (next to their output), so this default is a convenience for
# ad-hoc use, never the system of record.
DEFAULT_LEDGER_PATH = "~/.run-farm/rentals.jsonl"


class RentalLedger:
    """Append-only JSONL receipts of host rental outcomes (P9).

    Records `rented`, `running`, `destroyed` (carrying outcome + billed seconds +
    est cost), one JSON object per line -- logged as it happens, so a crash leaves
    a partial-but-honest record rather than nothing.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def default(cls) -> "RentalLedger":
        return cls(DEFAULT_LEDGER_PATH)

    def record(self, event: str, **fields) -> dict:
        rec = {"ts": round(time.time(), 3), "event": event, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
        return rec

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(ln) for ln in self.path.read_text().splitlines()
                if ln.strip()]

    def summary(self) -> dict:
        """Tally outcomes and total spend across all logged rentals.

        ⚠ Spend is summed over `destroyed` events only -- an in-flight rental's
        burn is NOT counted here, because est_cost is computed at teardown. A
        budget cap must project in-flight cost separately (see
        `run_farm.budget.CappedProvider`); do not treat this total as live spend.

        Note also that `by_outcome`'s `ok` means "the rental behaved" (host came
        up, work dispatched), NOT "the work succeeded" -- a leg whose command
        exits non-zero can still be `ok` here. Cost-per-successful-run needs the
        leg's own result, not this tally.
        """
        evs = self.events()
        destroyed = [e for e in evs if e["event"] == "destroyed"]
        by_outcome: dict[str, int] = {}
        for e in destroyed:
            o = e.get("outcome", "?")
            by_outcome[o] = by_outcome.get(o, 0) + 1
        return {
            "rentals": sum(1 for e in evs if e["event"] == "rented"),
            "by_outcome": by_outcome,
            "total_billed_min": round(sum(e.get("billed_s", 0)
                                          for e in destroyed) / 60, 1),
            "total_est_cost_usd": round(sum(e.get("est_cost_usd", 0)
                                            for e in destroyed), 6),
        }
