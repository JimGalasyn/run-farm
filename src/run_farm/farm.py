"""Governed fleet campaign: POLICY over the campaign MECHANISM.

The campaign boundary (driver/protocols) owns the *mechanism* — config-hashed
identity + idempotent skip (A/B), event streaming (C), fleet fan-out + preemption
recovery (D), host probing (E), cloud brokering (F). This module adds the
*policy* that makes a certificate lineage trustworthy, physics-agnostically:

  - envelope **preflight BEFORE rent** (don't pay for a config that can't hold);
  - a **launch-completeness gate** (the launched config-hash set must equal the
    planned one — no silently-dropped legs);
  - **shipment verification**: product-hash sidecars + worker **engine-SHA
    attestation** against a global tag (the host attests the blob it actually
    ran, never copied from the request);
  - a typed **cut-flow** so silent attrition is structurally impossible;
  - idempotent **corpus ingest** with conflict-never-overwrite.

Everything domain-specific is INJECTED: `preflight(cfg) -> [violations]`,
`stage_validate(plan, cfg) -> [violations]`, and `ingest(record)`. A caller with
no policy gets a thin, correct wrapper around `execute_config`.

The RunFn shipment contract (what a governed `run_fn` returns)::

    {"products": {name: <json-able record>, ...},
     "sidecar":  {name: sha256(record)},
     "attested_shas": {slot: <worker-measured blob sha>}}

receipts stream through `ctx.emit` as event records (P6: records, not fields).
"""
from __future__ import annotations

import hashlib
import json
from typing import Callable

from run_farm.driver import execute_config
from run_farm.reference import FileRunRegistry, JsonlEventSink
from run_farm.protocols import RunConfig

FLEET_STAGES = ["planned", "launch-gate", "completed", "shipped",
                "verified", "registered"]


def sha256_json(obj) -> str:
    """Canonical content hash: sorted keys, no whitespace, explicit UTF-8, and
    NaN/Inf rejected — so a sidecar hash compares byte-stably across hosts."""
    return hashlib.sha256(json.dumps(
        obj, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ensure_ascii=True).encode("utf-8")).hexdigest()


class CutFlow:
    """Per-leg accounting: every leg enters at stage 0; each stage records
    survivors; a drop carries a typed reason and forces every later stage
    False. `render()` shows the full waterfall — silent attrition is impossible."""

    def __init__(self, stages: list[str]):
        self.stages = stages
        self._idx = {s: i for i, s in enumerate(stages)}   # O(1) mark()
        self.rows: dict[str, list] = {}
        self.reasons: dict[tuple, str] = {}

    def enter(self, leg: str):
        self.rows[leg] = [None] * len(self.stages)
        for key in [k for k in self.reasons if k[0] == leg]:
            del self.reasons[key]           # a re-entered leg starts clean

    def mark(self, leg: str, stage: str, ok: bool, reason: str = ""):
        if leg not in self.rows:            # defensive: an unplanned/unknown leg
            self.enter(leg)                 # is entered rather than KeyError-ing
        i = self._idx[stage]
        self.rows[leg][i] = ok
        if not ok:
            self.reasons[(leg, stage)] = reason or "unspecified"
            for j in range(i + 1, len(self.stages)):
                self.rows[leg][j] = False
        else:
            self.reasons.pop((leg, stage), None)
            # a failed->ok transition un-forces the downstream Falses this
            # stage's old drop imposed: walk forward, resetting forced Falses
            # to None (unknown), stopping at the first stage with its OWN
            # recorded drop — it and everything it forces stay dropped
            for j in range(i + 1, len(self.stages)):
                if (leg, self.stages[j]) in self.reasons:
                    break
                if self.rows[leg][j] is False:
                    self.rows[leg][j] = None

    def table(self) -> dict:
        counts = [{"stage": s, "survivors": sum(1 for r in self.rows.values() if r[i])}
                  for i, s in enumerate(self.stages)]
        drops = [{"leg": leg, "stage": st, "reason": rs}
                 for (leg, st), rs in sorted(self.reasons.items())]
        return {"entered": len(self.rows), "waterfall": counts, "drops": drops}

    def render(self) -> str:
        t = self.table()
        out = [f"cut-flow: {t['entered']} legs entered"]
        out += [f"  {c['stage']:<24}{c['survivors']}" for c in t["waterfall"]]
        out += [f"  DROP {d['leg']} @ {d['stage']}: {d['reason']}" for d in t["drops"]]
        return "\n".join(out)


def leg_params(leg: dict, gtag: str, required_shas: dict, *,
               reserved: tuple[str, ...] = ()) -> dict:
    """Build the physics-agnostic `params` payload for a leg -> config.

    This is the half of leg->config that has nothing to do with any engine's config
    shape: the copy semantics and the unspoofable campaign-authoritative keys. The
    engine-specific half (which leg keys become top-level dataclass fields) lives in
    the `config_factory` that calls this.

    `reserved` names the EXTRA leg keys the caller's factory consumes as top-level
    fields, so they don't ALSO ride in params. Getting this set wrong changes
    params, which changes `config_hash`, which renames every run directory in the
    campaign -- so a factory that reads a key MUST reserve it, and any change to the
    set is a ledger-breaking change. Pin it with a golden test.
    """
    # cfg/plan are COPIED: config_hash() serializes params, so a caller mutating the
    # original leg dicts after plan() would silently change the config's identity
    # (run_name / registry dir / gate matching) and, when legs share one cfg object,
    # cross-contaminate each other.
    cfg = dict(leg["cfg"])
    base = ("rid", "cfg", "plan", "gtag", "required_shas")
    skip = set(base) | set(reserved)
    return {"rid": leg["rid"], "cfg": cfg, "plan": list(leg.get("plan", [])),
            **{k: leg[k] for k in leg if k not in skip},
            # per-leg COPY: mutating one config's required_shas must not alias into
            # other legs or the campaign's verification baseline
            "gtag": gtag, "required_shas": dict(required_shas)}


def simple_leg_to_config(leg: dict, gtag: str, required_shas: dict):
    """Default `config_factory`: a leg -> a `SimpleRunConfig`.

    The engine-agnostic factory. An engine with its own config shape supplies its
    own (see FarmCampaign's `config_factory` argument), reserving the leg keys it
    reads as top-level fields.
    """
    from run_farm.config import SimpleRunConfig
    return SimpleRunConfig(
        name=leg.get("name", "farm"),
        dtype=leg.get("dtype", "float32"),
        params=leg_params(leg, gtag, required_shas, reserved=("name", "dtype")))


def launch_gate(planned: list, launched: list) -> list[str]:
    """Refuse the WHOLE launch unless the launched config-hash set equals the
    planned one — a missing leg at launch (an argparse/config slip) is caught
    before any host is billed, not discovered in a short results table."""
    p = {c.config_hash(): c.params["rid"] for c in planned}
    q = {c.config_hash(): c.params["rid"] for c in launched}
    v = []
    if set(p) - set(q):
        v.append(f"MISSING legs at launch: {sorted(p[h] for h in set(p) - set(q))}")
    if set(q) - set(p):
        v.append(f"UNPLANNED legs at launch: {sorted(q[h] for h in set(q) - set(p))}")
    return v


def verify_shipment(shipment: dict, required_shas: dict) -> list[str]:
    """Product-hash sidecars + worker engine-SHA attestation vs a global tag."""
    v = []
    # standalone-utility shape guard: typed violations, never a raise. The RunFn
    # contract documents all three keys as PRESENT, so a missing key is
    # MALFORMED too — an incomplete shipment must not verify as valid
    if not isinstance(shipment, dict):
        return ["MALFORMED shipment: not a dict"]
    for key in ("products", "sidecar", "attested_shas"):
        if key not in shipment:
            v.append(f"MALFORMED {key}: missing")
        elif not isinstance(shipment[key], dict):
            v.append(f"MALFORMED {key}: not a dict")
    if v:
        return v
    for name, obj in shipment.get("products", {}).items():
        try:
            got = sha256_json(obj)
        except (ValueError, TypeError) as e:     # unhashable product (NaN, non-str keys)
            v.append(f"UNHASHABLE {name}: {e}")
            continue
        want = shipment.get("sidecar", {}).get(name)
        if want != got:
            v.append(f"HASH_MISMATCH {name}: sidecar {str(want)[:12]} != shipped {got[:12]}")
    for slot, sha in required_shas.items():
        att = shipment.get("attested_shas", {}).get(slot)
        if att != sha:
            v.append(f"SHA_MISMATCH {slot}: host ran {att}, tag requires {sha}")
    return v


class FarmCampaign:
    """Chamber-style policy for one fleet campaign: plan (pre-rent) -> gate ->
    execute via the mechanism -> verify -> ingest, over a typed cut-flow.

    Injected policy (all optional; omitting one makes that gate a no-op):
      preflight(cfg)         -> [violations]  (envelope walls, before rent)
      stage_validate(plan,cfg)-> [violations] (staging geometry admissibility)
      ingest(record)         -> None          (corpus persistence; default in-mem)

    The executor is campaign seam D: `execute_leg` runs the run_fn in-process
    (execute_config) by default; pass `run=` — a ZERO-ARG thunk returning one
    shipment dict (or None on idempotent skip) — to dispatch elsewhere (the caller
    adapts an Executor/ProviderExecutor.run into such a thunk). `execute_fleet` is
    the batch form that consumes the ProviderExecutor.run record list directly. All
    mechanism guarantees (A idempotent skip, B restart-after-death, C flushed
    events) are inherited unchanged."""

    def __init__(self, gtag: str, required_shas: dict, work_dir: str, *,
                 preflight: Callable[[dict], list] | None = None,
                 stage_validate: Callable[[list, dict], list] | None = None,
                 ingest: Callable[[dict], None] | None = None,
                 config_factory: Callable[[dict, str, dict], RunConfig]
                 = simple_leg_to_config):
        # own a copy: the verification baseline must not be mutable through the
        # caller's dict (or through any config.params aliasing it)
        self.gtag, self.required_shas = gtag, dict(required_shas)
        self.registry = FileRunRegistry(work_dir)        # mechanism A/B
        self.sink = JsonlEventSink()                     # mechanism C
        self.preflight = preflight or (lambda cfg: [])
        self.stage_validate = stage_validate or (lambda plan, cfg: [])
        self.ingest = ingest
        # leg -> RunConfig. Default builds a SimpleRunConfig; an engine with its own
        # config shape injects its own factory (reserving the leg keys it reads).
        self.config_factory = config_factory
        self.cf = CutFlow(FLEET_STAGES)
        self.configs: list = []
        self.ingested: dict = {}                         # rid -> content hash
        self._gated = False                              # gate_launch passed?

    def plan(self, legs: list[dict]) -> dict:
        self._gated = False                              # a new plan must re-gate
        seen, dups = set(), set()                        # O(n) duplicate scan
        for leg in legs:
            (dups if leg["rid"] in seen else seen).add(leg["rid"])
        if dups:                                         # a duplicate rid would
            raise ValueError(                            # conflate cut-flow rows
                f"duplicate rid(s) in plan: {sorted(dups)} — rids must be unique "
                "(cut-flow and ingest bookkeeping are keyed by rid)")
        ok = []
        for leg in legs:
            self.cf.enter(leg["rid"])
            pv = self.preflight(leg["cfg"]) + self.stage_validate(
                leg.get("plan", []), leg["cfg"])
            self.cf.mark(leg["rid"], "planned", not pv, "; ".join(pv))
            if not pv:
                ok.append(self.config_factory(leg, self.gtag, self.required_shas))
        self.configs = ok
        return {"planned": len(legs), "valid": len(ok)}

    def gate_launch(self, launched: list) -> list[str]:
        v = launch_gate(self.configs, launched)
        for c in self.configs:
            self.cf.mark(c.params["rid"], "launch-gate", not v, "; ".join(v) if v else "")
        self._gated = not v
        return v

    def execute_leg(self, config: RunConfig, run_fn, *, run=None) -> dict:
        rid = config.params["rid"]
        if rid not in self.cf.rows:                       # allow an unplanned config
            self.cf.enter(rid)
        runner = run or (lambda: execute_config(
            config, run_fn, registry=self.registry, sink=self.sink))
        try:
            shipment = runner()
        except Exception as e:                            # preemption / host death
            self.cf.mark(rid, "completed", False, f"HOST_LOST: {e}")
            return {"rid": rid, "status": "LOST", "reason": str(e)}
        if shipment is None:                              # idempotent skip (A)
            return self._on_skip(rid, config)
        return self._govern(rid, shipment)

    def execute_fleet(self, dispatch) -> dict:
        """Batch seam D: dispatch ALL planned+gated configs through an injected
        executor and govern each result (verify + ingest over the cut-flow).

        `dispatch(configs) -> [{"run", "result", "skipped"}, ...]` is the
        ProviderExecutor.run / remote run_one contract: `result` is the RunFn's
        shipment (or None when the worker's idempotent skip fired — then the
        DONE.json result is recovered and ingested, same as execute_leg, so the
        batch path never leaves the corpus short). Returns a dict of result-dicts
        (the same status dicts execute_leg/_govern return) keyed by rid for
        planned legs — plus, for protocol violations, entries keyed by the
        dispatch's unrecognized run string (REJECTED, never ingested).

        Launch-completeness is ENFORCED: raises RuntimeError unless a passing
        gate_launch() ran against the current plan (re-plan resets the gate)."""
        if not self._gated:
            raise RuntimeError(
                "execute_fleet before a passing gate_launch(): call plan(...) "
                "then gate_launch(configs) and resolve violations first")
        by_name = {c.run_name(): c for c in self.configs}
        out = {}
        for rec in dispatch(self.configs):
            c = by_name.get(rec.get("run"))
            if c is None:                                 # unplanned run name: a
                key = str(rec.get("run", "?"))            # protocol violation, never
                # collision-proof the key: an invalid record whose run string
                # equals a PLANNED rid must not reset that leg's cut-flow row or
                # shadow its real record as a "duplicate"
                planned_rids = {p.params["rid"] for p in self.configs}
                while key in planned_rids or key in out:
                    key += ".unplanned"
                self.cf.enter(key)                        # governed/ingested: never
                reason = "unplanned run name from dispatch (protocol violation)"
                self.cf.mark(key, "completed", False, reason)
                out[key] = {"rid": key, "status": "REJECTED", "violations": [reason]}
                continue
            rid = c.params["rid"]
            if rid in out:                                # duplicate record: keep the
                dup = f"{rid}.dup"                        # first, type the repeat
                self.cf.enter(dup)
                self.cf.mark(dup, "completed", False,
                             "duplicate dispatch record (protocol violation)")
                continue
            if rid not in self.cf.rows:
                self.cf.enter(rid)
            shipment = rec.get("result")
            if shipment is None:                          # idempotent skip / host lost
                if bool(rec.get("skipped")):
                    out[rid] = self._on_skip(rid, c)      # recover DONE.json + ingest
                else:
                    reason = "no result (host lost)"
                    self.cf.mark(rid, "completed", False, reason)
                    out[rid] = {"rid": rid, "status": "LOST", "reason": reason}
                continue
            out[rid] = self._govern(rid, shipment)
        # reconcile: a planned config the dispatch never reported = data loss
        for c in self.configs:
            rid = c.params["rid"]
            if rid not in out:
                reason = "dispatch returned no record for this leg (dropped?)"
                self.cf.mark(rid, "completed", False, reason)
                out[rid] = {"rid": rid, "status": "LOST", "reason": reason}
        return out

    def _on_skip(self, rid: str, config: RunConfig) -> dict:
        """Mechanism A idempotent skip: already-ingested -> SKIP_OK; otherwise
        recover the finished DONE.json result and govern it, so a skip still lands
        in the corpus and the cut-flow is never left silently unmarked."""
        if rid in self.ingested:
            # fully governed on its first pass — re-affirm every stage so a
            # terminal SKIP_OK never reads as silent attrition in the waterfall
            for s in ("completed", "shipped", "verified", "registered"):
                self.cf.mark(rid, s, True)
            return {"rid": rid, "status": "SKIP_OK", "reason": "already complete"}
        done = self.registry.register(config).dir / "DONE.json"
        if done.exists():
            return self._govern(rid, json.loads(done.read_text()))
        # a skip signal with NO recoverable DONE.json = the completion marker and
        # the result diverged (failed artifact sync, cross-registry mismatch, or a
        # lying injected runner). That is data loss, not success — surface it.
        reason = "skip signaled but no DONE.json result (possible data loss)"
        self.cf.mark(rid, "completed", False, reason)
        return {"rid": rid, "status": "LOST", "reason": reason}

    def _govern(self, rid: str, shipment) -> dict:
        """Verify + ingest one shipment over the cut-flow. A malformed shipment
        (not a dict, or missing `products`) is REJECTED with a typed violation at
        this policy boundary, never a KeyError."""
        self.cf.mark(rid, "completed", True)
        bad = None
        if not isinstance(shipment, dict) or not isinstance(shipment.get("products"), dict):
            bad = "malformed shipment (products missing or not a dict)"
        elif not isinstance(shipment.get("sidecar", {}), dict) or \
                not isinstance(shipment.get("attested_shas", {}), dict):
            bad = "malformed shipment (sidecar/attested_shas not dicts)"
        if bad:
            self.cf.mark(rid, "shipped", False, bad)
            return {"rid": rid, "status": "REJECTED", "violations": [bad]}
        self.cf.mark(rid, "shipped", True)
        v = verify_shipment(shipment, self.required_shas)
        self.cf.mark(rid, "verified", not v, "; ".join(v))
        if v:
            return {"rid": rid, "status": "REJECTED", "violations": v}
        return self._ingest(rid, shipment)

    def _ingest(self, rid: str, shipment: dict) -> dict:
        try:
            content = sha256_json(shipment["products"])
        except (ValueError, TypeError) as e:     # unhashable products -> typed reject
            self.cf.mark(rid, "registered", False, f"unhashable products: {e}")
            return {"rid": rid, "status": "REJECTED",
                    "violations": [f"unhashable products: {e}"]}
        if rid in self.ingested:
            if self.ingested[rid] == content:
                self.cf.mark(rid, "registered", True)   # terminal, not attrition
                return {"rid": rid, "status": "SKIP_OK",
                        "reason": "already in corpus, identical content"}
            # the original rid stays REGISTERED (its first, accepted content); the
            # rejected re-attempt is a distinct .conflict leg so neither is silent
            self.cf.mark(rid, "registered", True)
            self.cf.enter(rid + ".conflict")
            for s in FLEET_STAGES[:-1]:
                self.cf.mark(rid + ".conflict", s, True)
            self.cf.mark(rid + ".conflict", "registered", False,
                         "CONFLICT: rid in corpus with DIFFERENT content — never overwritten")
            return {"rid": rid, "status": "CONFLICT"}
        if self.ingest is not None:
            self.ingest(shipment["products"].get("record", shipment["products"]))
        self.ingested[rid] = content
        self.cf.mark(rid, "registered", True)
        return {"rid": rid, "status": "REGISTERED"}
