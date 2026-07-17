"""Cost as a first-class quantity: a pre-launch estimate and an ENFORCED cap.

A grant-funded campaign has a hard dollar ceiling, and "advisory" is not a ceiling.
Two pieces:

  estimate()       a pure pre-launch number: what a campaign shape will cost,
                   INCLUDING the failure tax (you pay for hosts that never boot).
  CappedProvider   a `Provider` decorator that REFUSES to rent past a dollar cap,
                   counting spend already booked PLUS in-flight burn.

`CappedProvider` wraps any `Provider` and is itself a `Provider`, so it composes at
the one seam money starts -- `rent()` -- with zero changes to any executor. Point a
Vast and a RunPod `CappedProvider` at ONE shared ledger and you get a single global
budget across providers, not one cap each.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from run_farm.protocols import HostSpec, LaunchSpec, Offer, RentedHost


class BudgetExceeded(RuntimeError):
    """A rent would push spend past the cap, so it was refused before any host was
    created. Distinct from the failover signals (`RentUnavailable`,
    `HostProbeFailed`): this is a deliberate stop, not a bad host -- it must halt the
    campaign, not fail over to the next offer."""


def estimate(n_runs: int, s_per_run: float, dph: float, *,
             n_hosts: int = 1, failure_tax: float = 0.0,
             acq_tax_usd: float = 0.0) -> dict:
    """Pre-launch cost of a campaign shape. Pure; call it before spending a cent.

    Args:
      n_runs       total runs (legs) to execute.
      s_per_run    wall-seconds of USEFUL work per run, on the host you'll rent.
      dph          dollars/hour of that host (all-in).
      n_hosts      how many rented hosts share the work (parallelism); only
                   affects wall-clock, not dollars.
      failure_tax  fraction ADDED to useful GPU-hours for hosts that boot and then
                   fail (a fractional overhead on the compute). Derive it from your
                   own `RentalLedger` history, not a guess.
      acq_tax_usd  FIXED dollars of failed rentals to acquire one host that boots.
                   Measured on Vast's cheap tier at ~$0.02-0.05; it is per host
                   acquired, NOT per run -- which is exactly why batching many runs
                   onto one host is the economical design (amortize it once, not
                   n_runs times). Charged n_hosts times here.

    Returns {gpu_h, wall_h, usd} -- gpu_h and usd include both taxes; wall_h
    assumes even split across n_hosts.
    """
    useful_gpu_h = n_runs * s_per_run / 3600.0
    gpu_h = useful_gpu_h * (1.0 + failure_tax)
    compute_usd = gpu_h * dph
    acq_usd = acq_tax_usd * max(1, n_hosts)
    return {
        "gpu_h": round(gpu_h, 4),
        "wall_h": round(gpu_h / max(1, n_hosts), 4),
        "usd": round(compute_usd + acq_usd, 4),
    }


def _in_flight_usd(ledger, now: float) -> float:
    """Dollars currently burning: rentals logged `rented` with no matching
    `destroyed`, charged at their dph for elapsed wall-time.

    ⚠ This is the piece `RentalLedger.summary()` does NOT capture -- it sums cost
    only at teardown. A cap that ignored in-flight burn would under-count exactly
    when a long rental is running, i.e. exactly when the cap matters most."""
    rented, destroyed = {}, set()
    for e in ledger.events():
        iid = e.get("instance_id")
        if iid is None:
            continue
        if e["event"] == "rented":
            rented[iid] = e
        elif e["event"] == "destroyed":
            destroyed.add(iid)
    total = 0.0
    for iid, e in rented.items():
        if iid in destroyed:
            continue
        dph = float(e.get("dph", 0) or 0)
        total += dph * max(0.0, now - float(e.get("ts", now))) / 3600.0
    return total


class CappedProvider:
    """A `Provider` that refuses to `rent` past a dollar cap. Wraps any Provider.

    Spend counted = booked (`ledger.summary()['total_est_cost_usd']`, teardown
    costs) + in-flight (open rentals at their dph x elapsed). If renting the next
    offer could not even begin without already being over, it raises
    `BudgetExceeded` BEFORE calling the inner `rent` -- so no host is created.

    It cannot know a rental's FINAL cost in advance (that depends on how long the
    work runs), so the guarantee is a pre-rent gate, not a hard mid-rental
    tripwire: a single rental can still overshoot by its own runtime. Size the cap
    with headroom for one host's max runtime, or pair it with a per-rental
    `rent_timeout`. The cap's job is to stop the CAMPAIGN, not to fractionally meter
    one box.
    """

    def __init__(self, inner, cap_usd: float, ledger, *,
                 clock: "callable" = time.time):
        self.inner = inner
        self.cap_usd = float(cap_usd)
        self.ledger = ledger
        self._clock = clock
        self.name = getattr(inner, "name", "capped")

    def spent_usd(self) -> float:
        """Booked + in-flight dollars, right now."""
        booked = self.ledger.summary().get("total_est_cost_usd", 0.0)
        return booked + _in_flight_usd(self.ledger, self._clock())

    def offers(self, spec: HostSpec) -> list[Offer]:
        return self.inner.offers(spec)          # listing is free; never gated

    @contextmanager
    def rent(self, offer: Offer, launch: LaunchSpec, *, timeout_s: float = 600):
        spent = self.spent_usd()
        if spent >= self.cap_usd:
            raise BudgetExceeded(
                f"cap ${self.cap_usd:.2f} reached (${spent:.4f} spent + in-flight) "
                f"-- refusing to rent offer {offer.id} @ ${offer.dph:.3f}/hr")
        with self.inner.rent(offer, launch, timeout_s=timeout_s) as host:
            yield host
