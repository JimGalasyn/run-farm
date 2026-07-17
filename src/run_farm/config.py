"""A batteries-included `RunConfig` + the registered-run helpers (contracts A/B).

Lifted wholesale from the source engine's `runs.py`: `config_hash`,
`save_checkpoint`, `load_checkpoint`, and `run_dir` were already physics-agnostic --
they hash a config, write a full-state `.npz`, and name a directory. Nothing about
them was ever soliton-specific, which is why they came out with the campaign layer
rather than staying behind.

`SimpleRunConfig` is the default for engines that don't need their own config shape.
Engines that do (a grid's N/L, a tissue's grid_size/verts_per_side) just satisfy the
`RunConfig` Protocol in `run_farm.protocols` and keep their own fields.

Checkpoints are `.npz` with the config embedded as JSON -- simple, dependency-light,
deterministic. Orbax replaces this layer when sharded multi-device arrays land (it
adds async + sharding-aware layout, not different semantics).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class SimpleRunConfig:
    """A `RunConfig` for engines that don't need their own config shape.

    Deliberately NOT a base class, and deliberately not sharing its hashing code
    with any engine's config: an engine's `to_json` bytes are the permanent names of
    every run directory it has ever written, so coupling them to THIS class's
    serialization would put every downstream ledger at the mercy of a run-farm
    refactor. Copy the four methods into your own config; don't inherit them. The
    duplication is the point.

    `name` is prefixed onto `run_name()` for legibility only -- identity is the hash.
    """

    name: str = "run"
    dtype: str = "float32"
    params: dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_json(self) -> str:
        # sort_keys is load-bearing, not cosmetic: a worker rebuilding params from
        # JSON gets whatever order the serializer emitted, and an order-sensitive
        # hash would fork one run's identity between driver and box.
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "SimpleRunConfig":
        return cls(**json.loads(s))

    def config_hash(self, n: int = 12) -> str:
        """Stable short hash for run-directory naming (mechanism A)."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:n]

    def run_name(self) -> str:
        return f"{self.name}_{self.config_hash()}"


def save_checkpoint(path, state: dict, config, step: int) -> None:
    """Write a full-state checkpoint: a flat dict of arrays (field, velocity,
    optimizer moments, RNG key, ...) + the RunConfig + the step counter."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"state__{k}": np.asarray(v) for k, v in state.items()}
    np.savez_compressed(path, __config__=config.to_json(), __step__=step,
                        **arrays)


def load_checkpoint(path, *, config_class=SimpleRunConfig) -> tuple[dict, Any, int]:
    """Read a checkpoint back: (state dict of jnp arrays, RunConfig, step).

    `config_class` is the concrete type to rebuild the embedded config with -- the
    farm holds a Protocol, and a Protocol cannot deserialize. Defaults to
    `SimpleRunConfig`; an engine with its own shape passes its own class (and
    typically wraps this in a one-line helper bound to it).

    Low blast radius by design: `reference.py` discards the returned config
    entirely and `store.py` never reads it, so for the farm's own paths this
    argument is inert. It matters only to callers who want their config back.
    """
    with np.load(path, allow_pickle=False) as d:
        config = config_class.from_json(str(d["__config__"]))
        step = int(d["__step__"])
        state = {k[len("state__"):]: jnp.asarray(d[k])
                 for k in d.files if k.startswith("state__")}
    return state, config, step


def run_dir(base, config) -> Path:
    """Config-hashed run directory with the config serialized into it, and a
    one-line entry appended to the base manifest (the run registry)."""
    base = Path(base)
    d = base / config.run_name()
    d.mkdir(parents=True, exist_ok=True)
    cfg_file = d / "config.json"
    if not cfg_file.exists():
        cfg_file.write_text(config.to_json() + "\n")
        with (base / "MANIFEST.jsonl").open("a") as mf:
            mf.write(json.dumps({"run": config.run_name(),
                                 "config": json.loads(config.to_json())})
                     + "\n")
    return d
