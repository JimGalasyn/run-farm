"""Shared core for the remote executors (Modal, Provider-over-SSH).

A closure can't cross a network, so the two things a remote worker needs that
can't be pickled travel by NAME:

    run_fn ref        ``pkg.mod:function``   -- the physics (the one injection)
    config-class ref  ``pkg.mod:ClassName``  -- the concrete RunConfig type

The second is forced by the Protocol: the farm types against `RunConfig`, which is
structural, and **a Protocol cannot deserialize**. The driver knows which concrete
config it planned with; the worker holds only bytes. So the type ships by name,
exactly as the RunFn does. Engines using `SimpleRunConfig` need not pass it.

The worker imports both, rebuilds a file-backed registry/sink at a shared-storage
path, and runs the same `execute_config` unit the in-process driver runs. Configs
travel as JSON (`RunConfig.to_json`), so nothing but stdlib + the already-installed
engine is needed on the far side.
"""

from __future__ import annotations

import importlib

from run_farm.driver import execute_config
from run_farm.reference import FileRunRegistry, JsonlEventSink

RunFnRef = str          # "module:function"
ConfigClassRef = str    # "module:ClassName"

DEFAULT_CONFIG_CLASS = "run_farm.config:SimpleRunConfig"


def _load_attr(ref: str, kind: str):
    if ":" not in ref:
        raise ValueError(f"{kind} ref must be 'module:name', got {ref!r}")
    module_name, attr = ref.split(":", 1)
    return getattr(importlib.import_module(module_name), attr)


def load_run_fn(ref: RunFnRef):
    """Import a `RunFn` from a ``'module:function'`` reference.

    The seam's one physics injection, made shippable: the local driver passes a
    callable, a remote worker passes this string and re-imports it on the box.
    """
    fn = _load_attr(ref, "run_fn")
    if not callable(fn):
        raise TypeError(f"{ref} is not callable")
    return fn


def load_config_class(ref: ConfigClassRef) -> type:
    """Import the concrete `RunConfig` class a worker rebuilds configs with.

    The Protocol's counterpart to `load_run_fn`: a Protocol is structural and has
    no `from_json` of its own to call, so the concrete type must cross the wire by
    name. Checked for `from_json` rather than for Protocol conformance because
    `issubclass()` against a data-member Protocol raises TypeError -- and because
    `from_json` is the only thing this function's caller actually needs.
    """
    cls = _load_attr(ref, "config_class")
    if not (isinstance(cls, type) and hasattr(cls, "from_json")):
        raise TypeError(f"{ref} is not a RunConfig class (no from_json)")
    return cls


def run_one(config_json: str, run_fn_ref: RunFnRef, work_dir: str, *,
            config_class: ConfigClassRef = DEFAULT_CONFIG_CLASS) -> dict:
    """Execute ONE config on this machine against a file registry at `work_dir`.

    The unit both remote workers run: deserialize the config, import the RunFn,
    build a `FileRunRegistry`/`JsonlEventSink` rooted at `work_dir` (a Modal
    Volume mount, or a rented box's disk synced back), and run `execute_config`.
    Returns ``{"run": <name>, "result": <record or None>, "skipped": bool}`` --
    small and JSON-safe, so it rides a Modal return value or an SSH stdout line.
    Full artifacts (checkpoints, events, triggered captures) stay in `work_dir`.
    """
    config = load_config_class(config_class).from_json(config_json)
    # Honor the config dtype on a fresh worker: jax defaults to x32, and x64
    # must be enabled before any array is created. In-process callers set this
    # themselves; a remote worker is a clean process, so do it here -- else a
    # float64 config silently runs in float32 on the box. This is the ONLY field
    # of a config the farm itself ever reads.
    if config.dtype == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
    run_fn = load_run_fn(run_fn_ref)
    registry = FileRunRegistry(work_dir)
    sink = JsonlEventSink()
    result = execute_config(config, run_fn, registry=registry, sink=sink)
    return {"run": config.run_name(), "result": result, "skipped": result is None}
