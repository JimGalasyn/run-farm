"""Provider diagnostics parity: what was learned vs what could not be looked at.

The failure being prevented: a monitor called `RunPodProvider.logs()`, which does not
exist, with stderr suppressed. It reported nothing, and nothing looked like a healthy
quiet host. These tests exist to keep "I could not look" from ever rendering as
"it is fine".
"""

from __future__ import annotations

from run_farm.diagnostics import Diagnostics, capabilities, collect, explain_failure


class _Pod:
    def __init__(self, pid):
        self.id = pid


class FullProvider:
    """A provider with the whole diagnostic surface (like the Vast adapter)."""

    name = "full"

    def __init__(self, *, dead=None, live=("h1",), logs="line one\nline two\n"):
        self._dead, self._live, self._logs = dead, live, logs

    def status(self, host_id):
        return {"id": host_id, "state": "running"}

    def dead_reason(self, host_id):
        return self._dead

    def logs(self, host_id, tail=2000):
        return self._logs

    def list_instances(self):
        return [_Pod(h) for h in self._live]

    def destroy(self, host_id):
        pass


class ThinProvider:
    """A provider with NO logs() -- the real RunPod shape."""

    name = "thin"

    def status(self, host_id):
        return {"id": host_id}

    def dead_reason(self, host_id):
        return None

    def list_instances(self):
        return [_Pod("h1")]

    def destroy(self, host_id):
        pass


# ------------------------------------------------------------- capabilities
def test_capabilities_reports_the_real_asymmetry():
    assert capabilities(FullProvider())["logs"] is True
    assert capabilities(ThinProvider())["logs"] is False


def test_capabilities_matches_the_shipped_providers():
    """Documents the live asymmetry that broke a monitor: the Vast adapter has
    logs(), the RunPod adapter does not. If that changes, this test says so rather
    than letting a caller discover it mid-campaign."""
    from run_farm.runpod import RunPodProvider
    from run_farm.vast import VastProvider

    assert capabilities(VastProvider)["logs"] is True
    assert capabilities(RunPodProvider)["logs"] is False
    for p in (VastProvider, RunPodProvider):               # parity where it exists
        caps = capabilities(p)
        assert caps["dead_reason"] and caps["destroy"] and caps["list_instances"]


# ------------------------------------------------------- gaps, not silence
def test_missing_capability_is_a_named_gap_not_an_empty_result():
    """THE test. `logs is None` alone is ambiguous; the gap disambiguates it."""
    d = collect(ThinProvider(), "h1")
    assert d.logs is None
    assert not d.complete, "a provider that cannot be asked is not a clean report"
    gap = next(g for g in d.gaps if g.field == "logs")
    assert gap.unsupported is True and "no logs()" in gap.reason
    assert "COULD NOT DETERMINE" in d.summary()


def test_a_complete_report_has_no_gaps_and_says_so():
    """Negative control for the above: with a full provider, empty means empty."""
    d = collect(FullProvider(), "h1")
    assert d.complete and d.gaps == ()
    assert d.alive is True and d.logs and "COULD NOT DETERMINE" not in d.summary()


def test_genuinely_absent_logs_differ_from_unaskable_logs():
    d = collect(FullProvider(logs=""), "h1")
    assert d.logs == "" and d.complete, "the host really produced nothing"


def test_a_raising_method_is_a_failure_gap_not_an_unsupported_one():
    """A method that exists but throws is a BUG, and must not be excused as a
    missing feature."""
    class Broken(FullProvider):
        def logs(self, host_id, tail=2000):
            raise RuntimeError("API 500")

    d = collect(Broken(), "h1")
    gap = next(g for g in d.gaps if g.field == "logs")
    assert gap.unsupported is False and "RuntimeError" in gap.reason


def test_collect_never_raises_even_when_everything_is_broken():
    """Diagnostics run on the error path. One that throws destroys the report you
    were trying to write."""
    class AllBroken:
        name = "broken"

        def status(self, h): raise RuntimeError("s")
        def dead_reason(self, h): raise RuntimeError("d")
        def logs(self, h, tail=2000): raise RuntimeError("l")
        def list_instances(self): raise RuntimeError("li")

    d = collect(AllBroken(), "h1")
    assert isinstance(d, Diagnostics) and len(d.gaps) == 4
    assert d.alive is None, "must not claim 'not alive' from a call that failed"


def test_alive_is_three_valued_and_never_guesses():
    """Reporting 'not alive' on the strength of a call that never happened is how a
    healthy host gets reaped."""
    assert collect(FullProvider(live=("h1",)), "h1").alive is True
    assert collect(FullProvider(live=()), "h1").alive is False

    class NoList(ThinProvider):
        list_instances = None                      # not callable -> unsupported

    d = collect(NoList(), "h1")
    assert d.alive is None
    assert any(g.field == "alive" and g.unsupported for g in d.gaps)


# ----------------------------------------------------------------- explain
def test_explain_distinguishes_host_death_from_work_failure():
    dead = explain_failure(FullProvider(dead="FAILED"), "h1", "rc=1: exit=1")
    assert "CONFIRMS the host died" in dead and "retryable" in dead

    live = explain_failure(FullProvider(dead=None, live=("h1",)), "h1", "rc=1")
    assert "genuine WORK failure" in live
    assert "fail the same way" in live, "must discourage a pointless re-rental"


def test_explain_lists_what_it_could_not_check():
    """The 2am log line must not imply completeness it does not have."""
    out = explain_failure(ThinProvider(), "h1", "rc=1: exit=1")
    assert "NOT CHECKED" in out and "logs" in out


def test_explain_refuses_to_conclude_when_liveness_is_unknown():
    class NoList(ThinProvider):
        list_instances = None

    out = explain_failure(NoList(), "h1")
    assert "do not conclude either way" in out
