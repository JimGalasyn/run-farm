"""The campaign.worker CLI: parses args, runs run_one, prints one RESULT line.
Exercised with a trivial injected RunFn (no GPU/physics)."""

import json
import sys

import run_farm.remote as remote
import run_farm.worker as worker
from run_farm import SimpleRunConfig as RunConfig


def _trivial(config, ctx):
    ctx.emit({"step": 0})
    return {"R": config.params["R"]}


def test_worker_main_prints_one_result_line(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(remote, "load_run_fn", lambda ref: _trivial)
    cfg = RunConfig("faddeev_cp1", params={"R": 2.0})
    monkeypatch.setattr(sys, "argv",
                        ["worker", "--config-json", cfg.to_json(),
                         "--run-fn", "x:y", "--work-dir", str(tmp_path)])
    worker.main()
    out = capsys.readouterr().out.strip().splitlines()
    line = next(l for l in out if l.startswith(worker.RESULT_PREFIX))
    rec = json.loads(line[len(worker.RESULT_PREFIX):])
    assert rec["run"] == cfg.run_name()
    assert rec["result"] == {"R": 2.0} and rec["skipped"] is False
