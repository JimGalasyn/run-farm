"""budget.estimate + CappedProvider -- the pre-launch number and the ENFORCED cap.

The cap's whole point is that "advisory" is not a cap: it must REFUSE to rent past
the ceiling, not warn. And it must count in-flight burn (open rentals at their dph x
elapsed), because RentalLedger.summary() sums cost only at teardown -- a cap that
ignored in-flight cost would under-count exactly when a long rental is running.
"""

import contextlib

import pytest

from run_farm.budget import BudgetExceeded, CappedProvider, estimate
from run_farm.ledger import RentalLedger
from run_farm.protocols import HostSpec, LaunchSpec, Offer

SPEC = HostSpec()
LAUNCH = LaunchSpec(image="img", onstart="x")


def _offer(oid="a", dph=0.30):
    return Offer(id=oid, dph=dph, gpu_name="RTX_4090", num_gpus=1,
                 reliability=0.99, inet_down_mbps=500, cuda_max=12.8)


# ----------------------------------------------------------------- estimate --

def test_estimate_useful_only():
    e = estimate(n_runs=100, s_per_run=36.0, dph=0.30)   # 100*36s = 1 gpu-h
    assert e["gpu_h"] == 1.0
    assert e["usd"] == 0.30
    assert e["wall_h"] == 1.0


def test_estimate_failure_tax_and_acquisition():
    e = estimate(n_runs=100, s_per_run=36.0, dph=0.30, n_hosts=10,
                 failure_tax=0.5, acq_tax_usd=0.04)
    assert e["gpu_h"] == 1.5                              # 1.0 * (1 + 0.5)
    assert e["wall_h"] == 0.15                            # split across 10 hosts
    assert e["usd"] == round(1.5 * 0.30 + 0.04 * 10, 4)   # compute + fixed acq tax


def test_acquisition_tax_is_per_host_not_per_run():
    """The measured shape: ~$0.04 to acquire one host that boots, amortized over
    however many runs share it -- which is the whole argument for batching."""
    one = estimate(n_runs=1000, s_per_run=19.0, dph=0.30, n_hosts=1, acq_tax_usd=0.04)
    many = estimate(n_runs=1000, s_per_run=19.0, dph=0.30, n_hosts=100, acq_tax_usd=0.04)
    assert one["usd"] < many["usd"]                       # more hosts = more acq tax
    assert round(many["usd"] - one["usd"], 4) == round(0.04 * 99, 4)


# ------------------------------------------------------------- CappedProvider --

class _FakeProvider:
    """Zero-spend provider: records a `rented` ledger row, yields, records
    `destroyed` with a fixed est cost -- mimicking the real teardown accounting."""
    name = "fake"

    def __init__(self, ledger, cost=0.10):
        self.ledger = ledger
        self.cost = cost
        self.rented_ids = []

    def offers(self, spec):
        return [_offer()]

    @contextlib.contextmanager
    def rent(self, offer, launch, *, timeout_s=600):
        self.rented_ids.append(offer.id)
        self.ledger.record("rented", instance_id=offer.id, dph=offer.dph)
        try:
            yield object()
        finally:
            self.ledger.record("destroyed", instance_id=offer.id,
                               outcome="ok", billed_s=100, est_cost_usd=self.cost)


def test_capped_provider_is_a_provider(tmp_path):
    from run_farm.protocols import Provider
    led = RentalLedger(tmp_path / "l.jsonl")
    cp = CappedProvider(_FakeProvider(led), cap_usd=1.0, ledger=led)
    assert isinstance(cp, Provider)                       # composes at the F seam
    assert cp.name == "fake"


def test_offers_never_gated(tmp_path):
    """Listing is free; only rent() spends, so only rent() is capped."""
    led = RentalLedger(tmp_path / "l.jsonl")
    cp = CappedProvider(_FakeProvider(led), cap_usd=0.0, ledger=led)
    assert cp.offers(SPEC)                                # not blocked despite $0 cap


def test_cap_refuses_when_booked_spend_exceeds(tmp_path):
    led = RentalLedger(tmp_path / "l.jsonl")
    inner = _FakeProvider(led, cost=0.60)
    cp = CappedProvider(inner, cap_usd=1.0, ledger=led)
    # two rents book $1.20 total; the third must be refused BEFORE inner.rent
    with cp.rent(_offer("a"), LAUNCH):
        pass
    with cp.rent(_offer("b"), LAUNCH):
        pass
    assert led.summary()["total_est_cost_usd"] == pytest.approx(1.20)
    with pytest.raises(BudgetExceeded):
        with cp.rent(_offer("c"), LAUNCH):
            pass
    assert "c" not in inner.rented_ids                    # never reached inner


def test_cap_counts_in_flight_burn(tmp_path):
    """A rental logged `rented` with no `destroyed` yet still counts against the
    cap at dph x elapsed -- the piece summary() misses. Drive a fake clock."""
    led = RentalLedger(tmp_path / "l.jsonl")
    # one open rental at $3.60/hr, "started" 1000s ago -> $1.00 in flight
    led.record("rented", instance_id="live", dph=3.60, ts=0.0)
    inner = _FakeProvider(led)
    cp = CappedProvider(inner, cap_usd=0.50, ledger=led, clock=lambda: 1000.0)
    assert cp.spent_usd() == pytest.approx(1.00, abs=1e-6)  # in-flight only
    with pytest.raises(BudgetExceeded):                     # already over on in-flight
        with cp.rent(_offer("x"), LAUNCH):
            pass
    assert "x" not in inner.rented_ids


def test_cap_allows_under_budget(tmp_path):
    led = RentalLedger(tmp_path / "l.jsonl")
    inner = _FakeProvider(led, cost=0.10)
    cp = CappedProvider(inner, cap_usd=1.0, ledger=led)
    with cp.rent(_offer("a"), LAUNCH):
        pass
    assert "a" in inner.rented_ids                        # under cap: proceeded
