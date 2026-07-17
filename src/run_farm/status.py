"""`campaign status`: a live operator view of a fleet run (#29).

During the 2026-06-15 session the operator had to grep driver logs and shell
`vastai` every few minutes to answer two questions: *what am I still paying
for?* and *how much has this cost?* This is that view in one command, read from
the two durable sources -- the Provider's live instance list (what is billing
right now) and the `VastLedger` (every rental, its outcome, and its spend):

    python -m run_farm.status --ledger output/eps_kick_fleet/vast_ledger.jsonl

Live (per-leg pending/running/done) state is the `list[LegResult]` a
`FleetExecutor.run()` returns; this view is the infrastructure half -- live
hosts + cumulative spend -- that outlives any one driver process.
"""

from __future__ import annotations

import argparse
from typing import Protocol, runtime_checkable

from run_farm.vast import VastLedger, VastProvider


@runtime_checkable
class InstanceLister(Protocol):
    """Any provider that can enumerate its live instances (cost-safety). The base
    `Provider` contract is just offers + rent; this is the optional extra
    `fleet_status` needs, so the annotation isn't pinned to `VastProvider` -- a
    `FakeProvider` or any other adapter exposing `list_instances` is accepted."""

    def list_instances(self) -> list: ...


def fleet_status(provider: InstanceLister | None = None,
                 ledger: VastLedger | None = None) -> dict:
    """Snapshot: instances live RIGHT NOW (cost-safety) + cumulative ledger
    spend/outcomes. Either source is optional -- pass a ledger for an after-the-
    fact cost report with no API call, or a provider for a pure live-instance
    check. The live list is the SAME v1 endpoint `rent()` verifies teardown
    against, so an instance here that no run owns is a leak to reap."""
    out: dict = {}
    if provider is not None:
        live = provider.list_instances()
        out["live"] = [{"id": i.id, "status": i.status, "dph": i.dph} for i in live]
        out["live_dph"] = round(sum(i.dph for i in live), 4)
    if ledger is not None:
        out["ledger"] = ledger.summary()
    return out


def _format(snap: dict) -> str:
    lines = []
    if "live" in snap:
        live = snap["live"]
        lines.append(f"LIVE: {len(live)} instance(s), "
                     f"${snap['live_dph']:.3f}/hr burning now")
        for i in live:
            lines.append(f"  {i['id']}  {i['status']:<10} ${i['dph']:.3f}/hr")
        if not live:
            lines.append("  (none -- nothing billing)")
    if "ledger" in snap:
        s = snap["ledger"]
        lines.append(f"LEDGER: {s['rentals']} rental(s), "
                     f"{s['total_billed_min']} min billed, "
                     f"${s['total_est_cost_usd']:.4f} spent")
        if s["by_outcome"]:
            outcomes = ", ".join(f"{k}={v}" for k, v in sorted(s["by_outcome"].items()))
            lines.append(f"  outcomes: {outcomes}")
    return "\n".join(lines) if lines else "(no provider or ledger given)"


def main(argv=None) -> None:  # pragma: no cover  (CLI glue)
    ap = argparse.ArgumentParser(description="Live fleet status: live instances + spend.")
    ap.add_argument("--ledger", help="path to a VastLedger jsonl (spend report)")
    ap.add_argument("--no-live", action="store_true",
                    help="skip the live-instance API call (ledger-only)")
    args = ap.parse_args(argv)
    provider = None if args.no_live else VastProvider()
    ledger = VastLedger(args.ledger) if args.ledger else None
    print(_format(fleet_status(provider, ledger)))


if __name__ == "__main__":  # pragma: no cover
    main()
