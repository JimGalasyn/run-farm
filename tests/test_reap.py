"""Reaper logic: ledger-diff, targeting, idempotent/classified destroy. No network."""
import json
import time


class _Inst:
    def __init__(self, id, status="running", dph=0.1, label=None, age_s=None):
        self.id = id; self.status = status; self.dph = dph
        self.raw = {}
        # `is not None` so an explicit empty label is preserved (mirrors the real
        # provider record), not dropped like a falsy value.
        if label is not None:
            self.raw["label"] = label
        if age_s is not None:                            # start_date = now - age
            self.raw["start_date"] = time.time() - age_s


class _FakeProvider:
    """list_instances + destroy. `errors` maps id -> list of exceptions raised on
    successive destroy calls (then it succeeds). `labels` / `ages` map id ->
    LaunchSpec label / age-in-seconds (for --label / --older-than). Records every
    attempt."""
    def __init__(self, ids, errors=None, labels=None, ages=None):
        labels = labels or {}; ages = ages or {}
        self._live = {i: _Inst(i, label=labels.get(i), age_s=ages.get(i))
                      for i in ids}
        self.destroyed = []
        self.attempts = []          # every destroy call (incl. failed)
        self.list_calls = 0
        self._errors = {k: list(v) for k, v in (errors or {}).items()}

    def list_instances(self):
        self.list_calls += 1
        return list(self._live.values())

    def destroy(self, iid):
        self.attempts.append(iid)
        q = self._errors.get(iid)
        if q:
            raise q.pop(0)
        self.destroyed.append(iid)
        self._live.pop(iid, None)


def _ledger(tmp_path, events):
    p = tmp_path / "vast_ledger.jsonl"
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return p


# -- ledger diff --------------------------------------------------------------
def test_leaked_ids_diffs_rented_minus_destroyed(tmp_path):
    from run_farm.reap import leaked_ids
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": 1},
        {"event": "running", "instance_id": 1},
        {"event": "rented", "instance_id": 2},
        {"event": "destroyed", "instance_id": 2, "verify": "gone"},
        {"event": "rented", "instance_id": 3},
    ])
    assert leaked_ids(led) == {"1", "3"}             # string ids (provider-agnostic)


def test_leaked_ids_missing_file_is_empty(tmp_path):
    from run_farm.reap import leaked_ids
    assert leaked_ids(tmp_path / "nope.jsonl") == set()


def test_leaked_ids_failed_teardown_stays_leaked(tmp_path):
    # a `destroyed` event clears a leak ONLY when verify=="gone" (issue #30/PR#22
    # review): a failed/unverified teardown must remain a suspect.
    from run_farm.reap import leaked_ids
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": 5},
        {"event": "destroyed", "instance_id": 5, "destroyed": False, "verify": "present"},
        {"event": "rented", "instance_id": 6},
        {"event": "destroyed", "instance_id": 6, "verify": "gone"},
    ])
    assert leaked_ids(led) == {"5"}                  # 5 failed teardown; 6 confirmed gone


def test_leaked_ids_keeps_string_ids(tmp_path):
    # ids are normalized to strings so the diff works for non-numeric (RunPod) ids
    # too; a ledger is single-provider, so there is no cross-provider mixing.
    from run_farm.reap import leaked_ids
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": "pod-abc"},  # RunPod-style string id
        {"event": "rented", "instance_id": 7},          # Vast-style numeric id
    ])
    assert leaked_ids(led) == {"pod-abc", "7"}


# -- targeting / scope --------------------------------------------------------
def test_reap_dry_run_destroys_nothing():
    from run_farm.reap import reap
    p = _FakeProvider([10, 11, 12])
    rep = reap(p, dry_run=True)
    assert rep["targeted"] == 3 and rep["destroyed"] == [] and p.destroyed == []


def test_reap_all_live_destroys_all():
    from run_farm.reap import reap
    p = _FakeProvider([10, 11, 12])
    rep = reap(p, dry_run=False)
    assert sorted(rep["destroyed"]) == [10, 11, 12] and sorted(p.destroyed) == [10, 11, 12]


def test_reap_ledger_scope_only_leaked_and_live(tmp_path):
    from run_farm.reap import reap
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": 1},
        {"event": "rented", "instance_id": 3},
        {"event": "destroyed", "instance_id": 3, "verify": "gone"},
    ])
    p = _FakeProvider([1, 99])                       # leaked={1,3}, live={1,99}
    rep = reap(p, ledger=led, dry_run=False)
    assert rep["destroyed"] == [1] and p.destroyed == [1]   # 99 untouched, 3 not live


def test_label_of_reads_raw_label():
    from run_farm.reap import _label_of
    assert _label_of(_Inst(1, label="farm-x")) == "farm-x"
    assert _label_of(_Inst(1)) is None                # no label stamped
    assert _label_of(object()) is None                # no `raw` attr -> robust


def test_reap_label_scope_only_matching_label():
    from run_farm.reap import reap
    p = _FakeProvider([10, 11, 12], labels={10: "farmA", 11: "farmA", 12: "farmB"})
    rep = reap(p, label="farmA", dry_run=False)
    assert rep["targeted"] == 2
    assert sorted(rep["destroyed"]) == [10, 11] and sorted(p.destroyed) == [10, 11]


def test_reap_label_and_ledger_intersect(tmp_path):
    # scopes are ANDed: leaked={10,11}, label farmA={10,12} -> only 10 reaped
    from run_farm.reap import reap
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": 10},
        {"event": "rented", "instance_id": 11},
    ])
    p = _FakeProvider([10, 11, 12], labels={10: "farmA", 11: "farmB", 12: "farmA"})
    rep = reap(p, ledger=led, label="farmA", dry_run=False)
    assert rep["destroyed"] == [10]


# -- age filter / helpers (issue #39) -----------------------------------------
def test_reap_older_than_keeps_only_aged():
    from run_farm.reap import reap
    p = _FakeProvider([10, 11], ages={10: 8 * 3600, 11: 600})   # 8h vs 10min
    rep = reap(p, older_than=6 * 3600, dry_run=False)
    assert rep["destroyed"] == [10]                  # young one spared


def test_reap_unknown_age_is_never_age_reaped():
    from run_farm.reap import reap
    p = _FakeProvider([10])                           # no age set -> unknown
    rep = reap(p, older_than=3600, dry_run=False)
    assert rep["targeted"] == 0 and rep["destroyed"] == []


def test_reap_label_and_older_than_intersect():
    from run_farm.reap import reap
    p = _FakeProvider([10, 11, 12],
                      labels={10: "f", 11: "f", 12: "g"},
                      ages={10: 8 * 3600, 11: 600, 12: 8 * 3600})
    # label f -> {10,11}; older 6h -> {10,12}; intersect -> {10}
    rep = reap(p, label="f", older_than=6 * 3600, dry_run=False)
    assert rep["destroyed"] == [10]


def test_parse_duration_units():
    from run_farm.reap import _parse_duration
    assert _parse_duration("90s") == 90
    assert _parse_duration("30m") == 1800
    assert _parse_duration("6h") == 21600
    assert _parse_duration("2d") == 172800
    assert _parse_duration("45") == 45             # bare number = seconds


def test_parse_duration_rejects_nonpositive():
    """A zero/negative duration would be a no-op age filter that still satisfies
    the --label gate -> the in-use footgun. Reject it (#39 review)."""
    import pytest
    from run_farm.reap import _parse_duration
    for bad in ("0", "-1h", "-30m", "0s"):
        with pytest.raises(ValueError):
            _parse_duration(bad)


def test_main_invalid_older_than_exits_clean(monkeypatch, capsys):
    from run_farm.reap import main
    _patch_provider(monkeypatch, _FakeProvider([10], labels={10: "f"}))
    # "0" is non-positive -> rejected (the `-1h` form argparse blocks even earlier
    # as a stray option, an extra safety layer); either way no destroy happens.
    rc = main(["--label", "f", "--older-than", "0", "--yes"])
    assert rc == 2 and "invalid --older-than" in capsys.readouterr().out


def test_instance_age_reads_start_or_none():
    from run_farm.reap import _instance_age_s
    assert _instance_age_s(_Inst(1, age_s=7200)) >= 7199        # ~2h
    assert _instance_age_s(_Inst(1)) is None                    # no start field
    assert _instance_age_s(object()) is None                    # no raw attr


def test_instance_age_naive_iso_is_utc():
    """A tz-naive ISO start time is read as UTC (machine-timezone stable)."""
    from run_farm.reap import _instance_age_s

    class I:
        raw = {"created_at": "2020-01-01T00:00:00"}              # naive -> UTC
    # one hour after that instant, age should be ~3600 regardless of $TZ
    assert abs(_instance_age_s(I(), now=1577836800 + 3600) - 3600) < 1


def test_reap_string_ids_ledger_scope(tmp_path):
    """RunPod-style non-numeric ids flow through the whole reap path (#45)."""
    from run_farm.reap import reap
    led = _ledger(tmp_path, [
        {"event": "rented", "instance_id": "pod-a"},
        {"event": "rented", "instance_id": "pod-b"},
        {"event": "destroyed", "instance_id": "pod-b", "verify": "gone"},
    ])
    p = _FakeProvider(["pod-a", "pod-c"])            # leaked={a,b}, live={a,c}
    rep = reap(p, ledger=led, dry_run=False)
    assert rep["destroyed"] == ["pod-a"]            # str id in report, pod-c untouched
    assert p.destroyed == ["pod-a"]


def test_make_provider_routes_by_name(monkeypatch):
    """_make_provider picks the adapter by name, lazily (no key needed in test)."""
    import pytest
    import run_farm.runpod as runpod
    import run_farm.vast as vast
    from run_farm import reap as reapmod
    monkeypatch.setattr(vast, "VastProvider", lambda: "VAST")
    monkeypatch.setattr(runpod, "RunPodProvider", lambda: "RUNPOD")
    assert reapmod._make_provider("vast") == "VAST"
    assert reapmod._make_provider("runpod") == "RUNPOD"
    with pytest.raises(ValueError, match="unknown --provider"):
        reapmod._make_provider("azure")


def test_main_provider_runpod_routes_via_factory(monkeypatch, tmp_path):
    """--provider runpod reaps through the (mocked) RunPod provider, string ids."""
    from run_farm import reap as reapmod
    led = _ledger(tmp_path, [{"event": "rented", "instance_id": "pod-a"}])
    p = _FakeProvider(["pod-a", "pod-z"])
    monkeypatch.setattr(reapmod, "_make_provider", lambda name: p)
    rc = reapmod.main(["--provider", "runpod", "--ledger", str(led), "--yes"])
    assert rc == 0 and p.destroyed == ["pod-a"]     # pod-z not leaked -> spared


def test_reap_reuses_prefetched_live_no_extra_list_call():
    from run_farm.reap import reap
    p = _FakeProvider([10])
    live = p.list_instances()                        # caller's single fetch
    assert p.list_calls == 1
    reap(p, dry_run=True, live=live)
    assert p.list_calls == 1                         # reap did not re-list (issue #30.2)


# -- classified / idempotent destroy (issue #30.1) ----------------------------
def test_reap_retries_transient_then_succeeds(monkeypatch):
    from run_farm import reap as reapmod
    from run_farm.reap import reap
    monkeypatch.setattr(reapmod.time, "sleep", lambda *_: None)   # no real backoff
    p = _FakeProvider([10], errors={10: [OSError("Temporary failure in name resolution")] * 2})
    rep = reap(p, dry_run=False, retries=4)
    assert rep["destroyed"] == [10] and rep["failed"] == []
    assert p.attempts.count(10) == 3                 # 2 transient fails + 1 success


def test_reap_already_gone_counts_as_success_not_failure():
    from run_farm.reap import reap
    p = _FakeProvider([10], errors={10: [RuntimeError(
        "DELETE /api/v0/instances/10/ -> HTTP 404: {'msg':'not found'}")]})
    rep = reap(p, dry_run=False, retries=4)
    assert rep["gone"] == [10] and rep["destroyed"] == [] and rep["failed"] == []
    assert p.attempts.count(10) == 1                 # no wasted backoff on already-gone


def test_reap_auth_error_fails_fast_no_retry():
    from run_farm.reap import reap
    p = _FakeProvider([10], errors={10: [RuntimeError(
        "DELETE /api/v0/instances/10/ -> HTTP 403: forbidden")] * 9})
    rep = reap(p, dry_run=False, retries=4)
    assert rep["failed"] == [10]
    assert p.attempts.count(10) == 1                 # terminal -> one attempt, no backoff


def test_reap_permanent_transient_failure_reported(monkeypatch):
    from run_farm import reap as reapmod
    from run_farm.reap import reap
    monkeypatch.setattr(reapmod.time, "sleep", lambda *_: None)   # no real backoff
    p = _FakeProvider([10], errors={10: [OSError("conn reset")] * 99})
    rep = reap(p, dry_run=False, retries=3)
    assert rep["failed"] == [10] and rep["destroyed"] == [] and rep["gone"] == []
    assert p.attempts.count(10) == 3                 # exhausted the retries


# -- main() CLI: --label scope + safety gate (issue #24) ----------------------
def _patch_provider(monkeypatch, provider):
    import run_farm.vast as vast
    monkeypatch.setattr(vast, "VastProvider", lambda: provider)


def test_main_unfiltered_label_destroy_refused(monkeypatch, capsys):
    """#39: --label with no orphan filter would kill in-use boxes -> refuse."""
    from run_farm.reap import main
    p = _FakeProvider([10, 11], labels={10: "farmA", 11: "farmB"})
    _patch_provider(monkeypatch, p)
    rc = main(["--label", "farmA", "--yes"])         # no --ledger/--older-than/--include
    assert rc == 2 and p.destroyed == []             # refused, nothing destroyed
    assert "REFUSING unfiltered --label destroy" in capsys.readouterr().out


def test_main_label_include_in_use_destroys_that_label(monkeypatch, capsys):
    """The explicit override destroys every box with the label (in-use included)."""
    from run_farm.reap import main
    p = _FakeProvider([10, 11], labels={10: "farmA", 11: "farmB"})
    _patch_provider(monkeypatch, p)
    rc = main(["--label", "farmA", "--include-in-use", "--yes"])
    assert rc == 0 and p.destroyed == [10]           # farmB (11) untouched
    out = capsys.readouterr().out
    assert "label 'farmA'" in out and "label='farmA'" in out


def test_main_label_older_than_destroys_only_old(monkeypatch, capsys):
    """--label + --older-than reaps only the aged (orphan-proxy) boxes -- the safe
    unattended path, no --include-in-use needed."""
    from run_farm.reap import main
    p = _FakeProvider([10, 11], labels={10: "farmA", 11: "farmA"},
                      ages={10: 8 * 3600, 11: 600})   # 10 is 8h old, 11 is 10min
    _patch_provider(monkeypatch, p)
    rc = main(["--label", "farmA", "--older-than", "6h", "--yes"])
    assert rc == 0 and p.destroyed == [10]           # young in-use box (11) spared
    assert "older-than 6h" in capsys.readouterr().out


def test_main_label_dry_run_lists_only(monkeypatch, capsys):
    from run_farm.reap import main
    p = _FakeProvider([10, 11], labels={10: "farmA", 11: "farmB"})
    _patch_provider(monkeypatch, p)
    rc = main(["--label", "farmA"])                  # no --yes
    assert rc == 0 and p.destroyed == []
    assert "DRY RUN" in capsys.readouterr().out


def test_main_empty_label_is_a_real_scope_not_all_account(monkeypatch, capsys):
    """`--label ''` must be honored as a scope (empty-labeled instances only) and
    displayed as such -- never silently reported/treated as the all-account nuke."""
    from run_farm.reap import main
    p = _FakeProvider([10, 11], labels={10: "", 11: "farmB"})
    _patch_provider(monkeypatch, p)
    rc = main(["--label", "", "--include-in-use", "--yes"])
    assert rc == 0 and p.destroyed == [10]           # only the empty-labeled one
    out = capsys.readouterr().out
    assert "label ''" in out and "ALL live instances" not in out


def test_main_unscoped_destroy_refused(monkeypatch, capsys):
    """--yes with no --ledger/--label/--all must refuse (won't nuke other sessions)."""
    from run_farm.reap import main
    p = _FakeProvider([10, 11])
    _patch_provider(monkeypatch, p)
    rc = main(["--yes"])
    assert rc == 2 and p.destroyed == []
    assert "REFUSING unscoped destroy" in capsys.readouterr().out


# -- structured classification + honest accounting (CM review #48) ------------
class _Coded(RuntimeError):
    """A provider error carrying a structured HTTP `.code` (like VastError)."""
    def __init__(self, code, msg="api error"):
        super().__init__(msg); self.code = code


def test_classify_prefers_structured_code_over_substring():
    from run_farm.reap import _classify
    assert _classify(_Coded(404)) == "gone"
    assert _classify(_Coded(410)) == "gone"
    assert _classify(_Coded(403)) == "auth"
    assert _classify(_Coded(500)) == "transient"
    # the crux: a "404" buried in a 500's message must NOT read as gone -- the
    # structured code wins, so a still-billing box is never called torn-down.
    assert _classify(_Coded(500, "req-404-deadbeef upstream timeout")) == "transient"


def test_classify_substring_fallback_when_no_code():
    from run_farm.reap import _classify
    assert _classify(RuntimeError("DELETE .../ -> HTTP 404: not found")) == "gone"
    assert _classify(RuntimeError("HTTP 403: forbidden")) == "auth"
    assert _classify(OSError("connection reset")) == "transient"


def test_classify_bug_type_is_terminal():
    from run_farm.reap import _classify
    assert _classify(TypeError("malformed record")) == "bug"
    assert _classify(KeyError("id")) == "bug"


def test_reap_bug_in_destroy_fails_fast_no_retry():
    """A TypeError (a real adapter bug) must NOT be retried as transient and then
    reported as a clean re-runnable 'failed' -- it won't self-heal (CM #48)."""
    from run_farm.reap import reap
    p = _FakeProvider([10], errors={10: [TypeError("malformed record")] * 9})
    rep = reap(p, dry_run=False, retries=4)
    assert rep["failed"] == [10] and p.attempts.count(10) == 1   # one shot, no backoff


def test_reap_dph_reclaimed_counts_only_cleared(monkeypatch):
    """A FAILED destroy leaves the box billing, so its dph must not inflate the
    reclaimed headline -- the number must be honest in the partial-failure run."""
    from run_farm import reap as reapmod
    from run_farm.reap import reap
    monkeypatch.setattr(reapmod.time, "sleep", lambda *_: None)
    p = _FakeProvider([10, 11], errors={11: [OSError("conn reset")] * 99})  # 11 fails
    rep = reap(p, dry_run=False, retries=2)
    assert rep["destroyed"] == [10] and rep["failed"] == [11]
    assert abs(rep["dph_reclaimed"] - 0.1) < 1e-9        # only 10's 0.1, not 0.2


def test_reap_dry_run_dph_is_potential():
    """Dry run reports the TARGETED dph (potential savings), nothing destroyed."""
    from run_farm.reap import reap
    rep = reap(_FakeProvider([10, 11]), dry_run=True)
    assert abs(rep["dph_reclaimed"] - 0.2) < 1e-9 and rep["destroyed"] == []


def test_parse_duration_rejects_nan():
    """'nan' parses to NaN, which `not secs > 0` catches -- lock it (CM #48)."""
    import pytest
    from run_farm.reap import _parse_duration
    with pytest.raises(ValueError):
        _parse_duration("nan")


def test_reap_relists_on_destructive_label_scope():
    """A real --label/--older-than destroy re-fetches live state so it can't act on
    a stale snapshot (a box recycled into the same label in the gap) -- TOCTOU #48."""
    from run_farm.reap import reap
    p = _FakeProvider([10, 11], labels={10: "f", 11: "f"})
    live = p.list_instances()                            # caller's snapshot
    assert p.list_calls == 1
    reap(p, label="f", dry_run=False, live=live)
    assert p.list_calls == 2                             # re-listed inside reap


def test_reap_no_relist_for_dry_run_or_unscoped():
    """Dry run and the (snapshot-safe) all-account scope reuse the passed listing."""
    from run_farm.reap import reap
    p = _FakeProvider([10], labels={10: "f"})
    live = p.list_instances()
    reap(p, label="f", dry_run=True, live=live)          # dry run -> no re-list
    reap(p, dry_run=False, live=live)                    # all-account -> no re-list
    assert p.list_calls == 1
