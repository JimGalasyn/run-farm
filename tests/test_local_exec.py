"""InProcessExecutor: the local machine as a run_multi participant. Exercised
with a trivial injected RunFn (no GPU/physics) so it's fast."""

import run_farm.remote as remote
from run_farm import run_multi, split_configs
from run_farm.local_exec import InProcessExecutor
from run_farm import SimpleRunConfig as RunConfig


def _trivial(config, ctx):
    ctx.emit({"step": 0, "R": config.params["R"]})
    return {"R": config.params["R"]}


def test_inprocess_runs_each_config(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "load_run_fn", lambda ref: _trivial)
    ex = InProcessExecutor("x:y", work_dir=str(tmp_path))
    assert ex.name == "local"
    cfgs = [RunConfig("faddeev_cp1", params={"R": 2.0}),
            RunConfig("faddeev_cp1", params={"R": 3.0})]
    recs = ex.run(cfgs)
    assert len(recs) == 2
    assert {r["result"]["R"] for r in recs} == {2.0, 3.0}
    assert all(not r["skipped"] for r in recs)
    # second run is the idempotent skip (DONE markers written under work_dir)
    assert all(r["skipped"] for r in ex.run(cfgs))


def test_inprocess_participates_in_run_multi(tmp_path, monkeypatch):
    """The local GPU merges into a multi-provider harvest like any cloud."""
    monkeypatch.setattr(remote, "load_run_fn", lambda ref: _trivial)
    ex = InProcessExecutor("x:y", work_dir=str(tmp_path))
    cfgs = [RunConfig("faddeev_cp1", params={"R": float(i)})
            for i in range(3)]
    report = run_multi(split_configs(cfgs, [ex]))
    assert report.ok and len(report.results) == 3
    assert all(r["provider"] == "local" for r in report.results.values())
