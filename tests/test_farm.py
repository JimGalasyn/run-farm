"""The governed campaign (campaign/farm.py): policy over the mechanism.

Physics-agnostic — a trivial mock RunFn drives plan -> gate -> execute -> verify
-> ingest so the POLICY (envelope preflight, launch-completeness gate, shipment
hash + engine-SHA attestation, restart-after-death, conflict guard, cut-flow) is
exercised without any model. Domain policy is injected (preflight/ingest).
"""
import pytest

from run_farm import FarmCampaign, launch_gate, verify_shipment
from run_farm.farm import sha256_json


def _record(rid, e=2.23):
    return {"rid": rid, "receipts": [{"type": "GAMMA", "e": e}]}


def _run_fn(behavior="good", die_flag=None):
    def run_fn(config, ctx):
        p = config.params
        if behavior == "die_once" and not die_flag.get("died"):
            die_flag["died"] = True
            ctx.emit({"kind": "progress", "step": 1})     # flushed (C)
            raise RuntimeError("host vanished mid-run")
        rec = _record(p["rid"])
        ctx.emit(rec["receipts"][0])                       # receipts = events
        products = {"record": rec}
        if behavior == "tamper":
            products = {"record": {**rec, "e": 9.99}}
        attested = dict(p["required_shas"])
        if behavior == "drift":
            attested[next(iter(attested))] = "drifted"
        return {"products": products, "sidecar": {"record": sha256_json(rec)},
                "attested_shas": attested}
    return run_fn


# in-envelope for the mock preflight below (a leg with C too large is dropped)
def _preflight(cfg):
    return [] if cfg.get("C", 0) <= 100 else [f"expulsion: C={cfg['C']} > 100"]


def _legs():
    good = {"L": 40.0, "dx": 0.8, "C": 50.0}
    legs = [{"rid": f"leg_{i}", "cfg": good, "seed": i} for i in range(4)]
    legs.append({"rid": "leg_bad", "cfg": {**good, "C": 400.0}, "seed": 9})
    return legs


def _camp(tmp_path, ingest=None):
    shas = {"engine": "aaaa111", "tracer": "c811b70"}
    return FarmCampaign("TAG", shas, str(tmp_path), preflight=_preflight,
                        ingest=ingest), shas


def test_preflight_drops_out_of_envelope_before_rent(tmp_path):
    camp, _ = _camp(tmp_path)
    st = camp.plan(_legs())
    assert st == {"planned": 5, "valid": 4}         # leg_bad dropped pre-rent
    assert {c.params["rid"] for c in camp.configs} == {"leg_0", "leg_1", "leg_2", "leg_3"}


def test_launch_gate_refuses_partial_launch(tmp_path):
    camp, _ = _camp(tmp_path)
    camp.plan(_legs())
    assert camp.gate_launch(camp.configs[:3])        # missing leg_3 -> refused
    assert camp.gate_launch(camp.configs) == []      # complete set passes


def test_execute_verify_ingest_and_idempotent_skip(tmp_path):
    ingested = []
    camp, _ = _camp(tmp_path, ingest=ingested.append)
    camp.plan(_legs())
    c0 = camp.configs[0]
    assert camp.execute_leg(c0, _run_fn("good"))["status"] == "REGISTERED"
    assert ingested == [_record("leg_0")]
    # re-execute the same config -> campaign idempotent skip (A)
    assert camp.execute_leg(c0, _run_fn("good"))["status"] == "SKIP_OK"


def test_restart_after_worker_death(tmp_path):
    camp, _ = _camp(tmp_path)
    camp.plan(_legs())
    c1 = camp.configs[1]
    flag = {}
    assert camp.execute_leg(c1, _run_fn("die_once", flag))["status"] == "LOST"
    assert (camp.registry.register(c1).dir / "events.jsonl").exists()   # C flushed
    assert camp.execute_leg(c1, _run_fn("die_once", flag))["status"] == "REGISTERED"


def test_shipment_hash_and_sha_attestation_reject(tmp_path):
    camp, _ = _camp(tmp_path)
    camp.plan(_legs())
    r = camp.execute_leg(camp.configs[2], _run_fn("tamper"))
    assert r["status"] == "REJECTED" and "HASH_MISMATCH" in r["violations"][0]
    r = camp.execute_leg(camp.configs[3], _run_fn("drift"))
    assert r["status"] == "REJECTED" and "SHA_MISMATCH" in r["violations"][0]


def test_conflict_never_overwritten(tmp_path):
    ingested = []
    camp, shas = _camp(tmp_path, ingest=ingested.append)
    camp.plan(_legs())
    camp.execute_leg(camp.configs[0], _run_fn("good"))
    r = camp._ingest("leg_0", {"products": {"record": _record("leg_0", e=9.99)},
                               "sidecar": {}, "attested_shas": shas})
    assert r["status"] == "CONFLICT"


def test_injected_executor_seam_d(tmp_path):
    """execute_leg accepts an injected `run` (seam D) instead of in-process."""
    camp, shas = _camp(tmp_path)
    camp.plan(_legs())
    c0 = camp.configs[0]
    fixed = {"products": {"record": _record("leg_0")},
             "sidecar": {"record": sha256_json(_record("leg_0"))},
             "attested_shas": shas}
    r = camp.execute_leg(c0, _run_fn("good"), run=lambda: fixed)
    assert r["status"] == "REGISTERED"


def test_execute_fleet_batch_dispatch(tmp_path):
    """Batch seam D: a mock dispatch (ProviderExecutor.run contract) returns a
    good leg, a tampered leg, and a skipped leg; execute_fleet governs each."""
    ingested = []
    camp, shas = _camp(tmp_path, ingest=ingested.append)
    camp.plan(_legs())                                    # 4 valid configs
    assert camp.gate_launch(camp.configs) == []           # execute_fleet enforces this
    # make leg_1's skip GENUINE: run it once so its result is ingested — a skip
    # record with no recoverable result is LOST (data loss), not SKIP_OK
    assert camp.execute_leg(camp.configs[1], _run_fn("good"))["status"] == "REGISTERED"

    def dispatch(configs):
        recs = []
        for i, c in enumerate(configs):
            rid = c.params["rid"]
            rec = _record(rid)
            if i == 1:                                    # skipped (worker idempotent skip)
                recs.append({"run": c.run_name(), "result": None, "skipped": True})
            elif i == 2:                                  # tampered product
                bad = {**rec, "e": 9.99}
                recs.append({"run": c.run_name(),
                             "result": {"products": {"record": bad},
                                        "sidecar": {"record": sha256_json(rec)},
                                        "attested_shas": dict(shas)},
                             "skipped": False})
            else:
                recs.append({"run": c.run_name(),
                             "result": {"products": {"record": rec},
                                        "sidecar": {"record": sha256_json(rec)},
                                        "attested_shas": dict(shas)},
                             "skipped": False})
        return recs

    out = camp.execute_fleet(dispatch)
    assert out["leg_0"]["status"] == "REGISTERED"
    assert out["leg_1"]["status"] == "SKIP_OK"
    assert out["leg_2"]["status"] == "REJECTED"
    assert out["leg_3"]["status"] == "REGISTERED"
    assert [r["rid"] for r in ingested] == ["leg_1", "leg_0", "leg_3"]


def test_skip_without_done_is_lost(tmp_path):
    """A skip signal with no recoverable DONE.json = data loss -> LOST, and the
    cut-flow records the drop (never a silent SKIP_OK)."""
    camp, _ = _camp(tmp_path)
    camp.plan(_legs())
    r = camp.execute_leg(camp.configs[0], _run_fn("good"), run=lambda: None)
    assert r["status"] == "LOST" and "no DONE.json" in r["reason"]


def test_leg_cannot_spoof_campaign_identity(tmp_path):
    """A leg carrying gtag/required_shas keys must NOT override the campaign's —
    the attestation identity is campaign-authoritative."""
    from run_farm import simple_leg_to_config as leg_to_config
    leg = {"rid": "spoof", "cfg": {"L": 40.0, "dx": 0.8},
           "gtag": "EVIL", "required_shas": {"engine": "spoofed"}}
    c = leg_to_config(leg, "REAL-TAG", {"engine": "aaaa111"})
    assert c.params["gtag"] == "REAL-TAG"
    assert c.params["required_shas"] == {"engine": "aaaa111"}


def test_execute_fleet_requires_passing_gate(tmp_path):
    """execute_fleet refuses to dispatch before a passing gate_launch; a
    re-plan resets the gate."""
    camp, _ = _camp(tmp_path)
    camp.plan(_legs())
    with pytest.raises(RuntimeError, match="gate_launch"):
        camp.execute_fleet(lambda configs: [])
    assert camp.gate_launch(camp.configs) == []
    camp.execute_fleet(lambda configs: [])                 # gated: dispatches fine
    camp.plan(_legs())                                     # re-plan resets the gate
    with pytest.raises(RuntimeError, match="gate_launch"):
        camp.execute_fleet(lambda configs: [])


def test_required_shas_not_aliased(tmp_path):
    """Mutating the caller's dict or a config's params must not move the
    campaign's verification baseline (attestation identity is stable)."""
    from run_farm import simple_leg_to_config as leg_to_config
    caller = {"engine": "aaaa111"}
    camp = FarmCampaign("TAG", caller, str(tmp_path), preflight=_preflight)
    caller["engine"] = "mutated-by-caller"
    assert camp.required_shas == {"engine": "aaaa111"}
    c = leg_to_config({"rid": "x", "cfg": {"L": 40.0, "dx": 0.8}},
                      "TAG", camp.required_shas)
    c.params["required_shas"]["engine"] = "mutated-via-config"
    assert camp.required_shas == {"engine": "aaaa111"}


def test_cutflow_reenter_clears_stale_reasons(tmp_path):
    """Re-entering a leg (re-plan) clears its old drop reasons — no stale drops
    reported for a reset row."""
    from run_farm import CutFlow
    cf = CutFlow(["a", "b"])
    cf.enter("leg"); cf.mark("leg", "a", False, "old failure")
    cf.enter("leg")
    assert not cf.table()["drops"]


def test_execute_fleet_reconciles_dispatch(tmp_path):
    """A dispatch that drops a leg, duplicates one, or invents an unplanned run
    name is reconciled: missing -> LOST, duplicate -> first kept + typed drop,
    unknown -> REJECTED (never governed/ingested)."""
    ingested = []
    camp, shas = _camp(tmp_path, ingest=ingested.append)
    camp.plan(_legs())
    assert camp.gate_launch(camp.configs) == []

    def dispatch(configs):
        good = lambda c: {"run": c.run_name(),
                          "result": {"products": {"record": _record(c.params["rid"])},
                                     "sidecar": {"record": sha256_json(_record(c.params["rid"]))},
                                     "attested_shas": dict(shas)},
                          "skipped": False}
        recs = [good(configs[0]), good(configs[0])]        # duplicate leg_0
        recs.append({"run": "ghost_run", "result": {"products": {"record": {"x": 1}}},
                     "skipped": False})                    # unplanned name
        recs.append(good(configs[1]))
        return recs                                        # legs 2,3 dropped

    out = camp.execute_fleet(dispatch)
    assert out["leg_0"]["status"] == "REGISTERED"
    assert out["leg_1"]["status"] == "REGISTERED"
    assert out["ghost_run"]["status"] == "REJECTED"
    assert out["leg_2"]["status"] == "LOST" and "no record" in out["leg_2"]["reason"]
    assert out["leg_3"]["status"] == "LOST"
    assert [r["rid"] for r in ingested] == ["leg_0", "leg_1"]   # ghost never ingested
    drops = {d["leg"]: d["reason"] for d in camp.cf.table()["drops"]}
    assert "leg_0.dup" in drops and "duplicate" in drops["leg_0.dup"]


def test_unplanned_name_cannot_shadow_planned_leg(tmp_path):
    """HIGH-PRI: an invalid dispatch record whose run string equals a PLANNED
    rid must not reset that leg's cut-flow row or block its real record — the
    violation key is disambiguated (.unplanned) and the real leg still lands."""
    ingested = []
    camp, shas = _camp(tmp_path, ingest=ingested.append)
    camp.plan(_legs())
    assert camp.gate_launch(camp.configs) == []

    def dispatch(configs):
        good = lambda c: {"run": c.run_name(),
                          "result": {"products": {"record": _record(c.params["rid"])},
                                     "sidecar": {"record": sha256_json(_record(c.params["rid"]))},
                                     "attested_shas": dict(shas)},
                          "skipped": False}
        # malicious/buggy record: its run string IS a planned rid (not a run_name)
        ghost = {"run": "leg_0", "result": {"products": {"record": {"x": 1}}},
                 "skipped": False}
        return [ghost] + [good(c) for c in configs]

    out = camp.execute_fleet(dispatch)
    assert out["leg_0"]["status"] == "REGISTERED"          # real leg NOT shadowed
    assert out["leg_0.unplanned"]["status"] == "REJECTED"  # ghost disambiguated
    assert "leg_0" in [r["rid"] for r in ingested]
    # leg_0's cut-flow row is fully governed, not reset by the ghost
    assert all(v is True for v in camp.cf.rows["leg_0"])


def test_verify_shipment_standalone_shape_guard():
    """Exported verify_shipment self-guards malformed shapes with typed
    violations instead of raising — and a MISSING contract key is MALFORMED,
    never silently valid."""
    assert "MALFORMED" in verify_shipment({"products": None}, {})[0]
    assert "MALFORMED" in verify_shipment({"products": {}, "sidecar": None}, {})[0]
    assert "MALFORMED" in verify_shipment("not-a-dict", {})[0]
    v = verify_shipment({}, {})                     # all three contract keys missing
    assert len(v) == 3 and all("missing" in x for x in v)
    v = verify_shipment({"products": {}, "sidecar": {}}, {})
    assert v == ["MALFORMED attested_shas: missing"]


def test_plan_rejects_duplicate_rids(tmp_path):
    """Duplicate rids would conflate cut-flow rows and ingest bookkeeping —
    rejected at plan-time with a clear error."""
    camp, _ = _camp(tmp_path)
    legs = _legs()[:2]
    legs[1] = dict(legs[1], rid=legs[0]["rid"])
    with pytest.raises(ValueError, match="duplicate rid"):
        camp.plan(legs)


def test_leg_cfg_and_plan_copied_for_identity_stability(tmp_path):
    """cfg/plan feed RunConfig.config_hash(); mutating the caller's dicts after
    planning must not change a config's identity or leak across legs sharing
    one cfg object."""
    from run_farm import simple_leg_to_config as leg_to_config
    shared_cfg = {"L": 40.0, "dx": 0.8, "C": 50.0}
    shared_plan = [{"oid": "a"}]
    c1 = leg_to_config({"rid": "l1", "cfg": shared_cfg, "plan": shared_plan}, "T", {})
    c2 = leg_to_config({"rid": "l2", "cfg": shared_cfg, "plan": shared_plan}, "T", {})
    h1, h2 = c1.config_hash(), c2.config_hash()
    shared_cfg["C"] = 9999.0                       # caller mutates after planning
    shared_plan.append({"oid": "b"})
    assert c1.params["cfg"]["C"] == 50.0 and len(c1.params["plan"]) == 1
    assert (c1.config_hash(), c2.config_hash()) == (h1, h2)   # identity stable
    c1.params["cfg"]["C"] = 123.0                  # nor can one leg leak to another
    assert c2.params["cfg"]["C"] == 50.0


def test_malformed_shipment_rejected_not_crash(tmp_path):
    """A RunFn returning a shipment with no `products` is REJECTED at the policy
    boundary with a typed violation, never a KeyError."""
    camp, _ = _camp(tmp_path)
    camp.plan(_legs())

    def bad_run_fn(config, ctx):
        return {"sidecar": {}, "attested_shas": {}}       # missing "products"
    r = camp.execute_leg(camp.configs[0], bad_run_fn)
    assert r["status"] == "REJECTED" and "malformed" in r["violations"][0]


def test_execute_leg_unplanned_config_no_keyerror(tmp_path):
    """execute_leg on a config that wasn't plan()-entered auto-enters it in the
    cut-flow rather than KeyError-ing."""
    camp, shas = _camp(tmp_path)
    camp.plan(_legs())
    unplanned = camp.configs[0]
    camp.cf.rows.clear()                                   # simulate 'never planned'
    r = camp.execute_leg(unplanned, _run_fn("good"))
    assert r["status"] == "REGISTERED"
    assert unplanned.params["rid"] in camp.cf.rows


def test_unhashable_product_rejected_not_crash(tmp_path):
    """A product carrying NaN (rejected by the canonical hash's allow_nan=False)
    is REJECTED at the boundary, not raised through execute_leg."""
    camp, shas = _camp(tmp_path)
    camp.plan(_legs())

    def nan_run_fn(config, ctx):
        rec = {"rid": config.params["rid"], "e": float("nan")}
        return {"products": {"record": rec}, "sidecar": {},
                "attested_shas": dict(shas)}
    r = camp.execute_leg(camp.configs[0], nan_run_fn)
    assert r["status"] == "REJECTED" and "UNHASHABLE" in r["violations"][0]


def test_none_products_rejected_not_crash(tmp_path):
    """{'products': None} (or sidecar None) is a typed REJECT, not a NoneType crash."""
    camp, shas = _camp(tmp_path)
    camp.plan(_legs())
    r = camp.execute_leg(camp.configs[0],
                         lambda config, ctx: {"products": None, "sidecar": {},
                                              "attested_shas": dict(shas)})
    assert r["status"] == "REJECTED" and "malformed" in r["violations"][0]
    r = camp.execute_leg(camp.configs[1],
                         lambda config, ctx: {"products": {"record": {"x": 1}},
                                              "sidecar": None,
                                              "attested_shas": dict(shas)})
    assert r["status"] == "REJECTED" and "malformed" in r["violations"][0]


def test_skip_ok_leaves_no_attrition_in_cutflow(tmp_path):
    """A terminal SKIP_OK re-execution marks every stage — the waterfall shows
    the leg fully governed, not silently dropped."""
    camp, _ = _camp(tmp_path)
    camp.plan(_legs())
    assert camp.gate_launch(camp.configs) == []
    c0 = camp.configs[0]
    assert camp.execute_leg(c0, _run_fn("good"))["status"] == "REGISTERED"
    assert camp.execute_leg(c0, _run_fn("good"))["status"] == "SKIP_OK"
    rid = c0.params["rid"]
    assert all(v is True for v in camp.cf.rows[rid]), camp.cf.rows[rid]


def test_verify_shipment_and_launch_gate_units(tmp_path):
    assert verify_shipment({"products": {}, "sidecar": {}, "attested_shas": {}}, {}) == []
    v = verify_shipment({"products": {"r": {"x": 1}}, "sidecar": {"r": "nope"},
                         "attested_shas": {}}, {})
    assert v and "HASH_MISMATCH" in v[0]
    # launch_gate as a standalone export: missing and unplanned both flagged
    camp, _ = _camp(tmp_path)
    camp.plan(_legs())
    cfgs = camp.configs
    assert launch_gate(cfgs, cfgs) == []
    assert "MISSING" in launch_gate(cfgs, cfgs[:2])[0]
    assert "UNPLANNED" in launch_gate(cfgs[:2], cfgs)[0]


def test_cutflow_refail_unforces_downstream(tmp_path):
    """HIGH-PRI: a failed->ok re-mark (e.g. gate_launch re-run after a fix) must
    un-force the downstream Falses its old drop imposed — otherwise later stages
    show 0 survivors with no drop reason (silent attrition). A downstream stage
    with its OWN drop stays dropped, as does everything it forces."""
    from run_farm import CutFlow
    cf = CutFlow(["a", "b", "c", "d"])
    cf.enter("leg")
    cf.mark("leg", "a", False, "incomplete launch")
    assert cf.rows["leg"] == [False, False, False, False]
    cf.mark("leg", "a", True)                     # fixed and re-gated
    assert cf.rows["leg"] == [True, None, None, None]
    assert not cf.table()["drops"]
    # mixed: a genuine downstream drop survives the upstream fix
    cf2 = CutFlow(["a", "b", "c", "d"])
    cf2.enter("leg")
    cf2.mark("leg", "c", False, "genuine c-failure")   # forces d False
    cf2.mark("leg", "a", False, "incomplete launch")   # forces b,c,d False
    cf2.mark("leg", "a", True)
    assert cf2.rows["leg"] == [True, None, False, False]
    assert {d["stage"] for d in cf2.table()["drops"]} == {"c"}
