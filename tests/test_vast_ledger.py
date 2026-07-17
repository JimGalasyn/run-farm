"""Offline tests for the Vast ledger (file I/O only — no network, no API key).

The live VastClient is network/spend-bound and not unit-tested here; VastLedger
is pure JSONL accounting and IS, so the receipts logic (outcomes + cost tally)
has a regression guard.
"""

from run_farm.vast import VastLedger


def test_vast_ledger_records_and_summarizes(tmp_path):
    led = VastLedger(tmp_path / "vast-ledger.jsonl")
    assert led.events() == []                      # nothing yet

    # a host that failed its probe, then a host that ran to completion
    led.record("rented", offer_id=1, instance_id=10, dph=0.15, gpu="RTX 3090")
    led.record("destroyed", offer_id=1, instance_id=10, outcome="host_failed",
               billed_s=90, est_cost_usd=0.00375)
    led.record("rented", offer_id=2, instance_id=11, dph=0.15, gpu="RTX 3090")
    led.record("running", offer_id=2, instance_id=11, provision_s=120)
    led.record("destroyed", offer_id=2, instance_id=11, outcome="ok",
               billed_s=900, est_cost_usd=0.0375)

    evs = led.events()
    assert len(evs) == 5
    assert all("ts" in e and "event" in e for e in evs)   # every line stamped

    s = led.summary()
    assert s["rentals"] == 2
    assert s["by_outcome"] == {"host_failed": 1, "ok": 1}
    assert abs(s["total_est_cost_usd"] - 0.04125) < 1e-9   # 0.00375 + 0.0375
    assert abs(s["total_billed_min"] - 16.5) < 1e-9        # (90 + 900) / 60


def test_vast_ledger_is_append_only(tmp_path):
    p = tmp_path / "led.jsonl"
    VastLedger(p).record("rented", offer_id=1)
    VastLedger(p).record("rented", offer_id=2)              # reopen, append
    assert [e["offer_id"] for e in VastLedger(p).events()] == [1, 2]
