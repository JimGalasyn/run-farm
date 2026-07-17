"""The shipped physics-free RunFns: they must honor the RunFn contract so a fleet
smoke test with no engine actually exercises A/B/C.

Driven through the real driver + file registry (not called bare), because the
point of these is that `execute_config` treats them exactly like a physics RunFn.
"""

import pytest

from run_farm import SimpleRunConfig
from run_farm.driver import execute_config
from run_farm.reference import FileRunRegistry, JsonlEventSink
from run_farm.testing import counting_run_fn, echo_run_fn, failing_run_fn


def _reg_sink(tmp_path):
    return FileRunRegistry(str(tmp_path)), JsonlEventSink()


def test_echo_run_fn_registers_and_finishes(tmp_path):
    reg, sink = _reg_sink(tmp_path)
    cfg = SimpleRunConfig(name="e", params={"i": 1})
    out = execute_config(cfg, echo_run_fn, registry=reg, sink=sink)
    assert out["ok"] is True and out["resumed_from"] is None
    assert reg.is_complete(reg.register(cfg))
    assert (tmp_path / cfg.run_name() / "events.jsonl").exists()


def test_counting_run_fn_reaches_target(tmp_path):
    reg, sink = _reg_sink(tmp_path)
    cfg = SimpleRunConfig(name="c", params={"steps": 5})
    out = execute_config(cfg, counting_run_fn, registry=reg, sink=sink)
    assert out["final_count"] == 5.0


def test_counting_run_fn_resumes_not_restarts(tmp_path):
    """Contract B without physics: after a checkpoint at step 2, a resumed run must
    CONTINUE the count to the target, not start over."""
    reg, sink = _reg_sink(tmp_path)
    cfg = SimpleRunConfig(name="c", params={"steps": 5})
    h = reg.register(cfg)
    # simulate a preemption: run the first 2 steps by hand, checkpointing state
    import numpy as np
    reg.save(h, {"count": np.asarray([2.0])}, step=2)
    # now resume: load feeds ctx.resume, and the fn must go 3,4,5 -> 5.0
    out = execute_config(cfg, counting_run_fn, registry=reg, sink=sink)
    assert out["final_count"] == 5.0


def test_failing_run_fn_raises_and_does_not_finish(tmp_path):
    reg, sink = _reg_sink(tmp_path)
    cfg = SimpleRunConfig(name="f", params={})
    with pytest.raises(RuntimeError, match="simulated crash"):
        execute_config(cfg, failing_run_fn, registry=reg, sink=sink)
    # a crashed run must NOT be marked complete (so re-submit runs it again)
    assert not reg.is_complete(reg.register(cfg))
