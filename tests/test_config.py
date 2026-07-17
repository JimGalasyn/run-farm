"""SimpleRunConfig + the A/B checkpoint helpers, and the RunConfig Protocol edges.

The load-bearing properties: SimpleRunConfig satisfies the structural Protocol,
config_hash is stable and order-insensitive, the checkpoint round-trip carries full
state at fixed dtype, and -- the documented contract edge -- from_json is lossy in
TYPE but not in HASH (a nested dataclass comes back a dict, a tuple comes back a
list, and neither changes the identity). That last one is pinned deliberately:
it's a local/remote divergence that would otherwise only surface on a rented box.
"""

import dataclasses

import numpy as np

from run_farm.config import (SimpleRunConfig, load_checkpoint, run_dir,
                             save_checkpoint)
from run_farm.protocols import RunConfig


def test_simple_run_config_satisfies_protocol():
    c = SimpleRunConfig(name="demo", params={"i": 1})
    assert isinstance(c, RunConfig)                       # presence-only, structural
    assert c.dtype == "float32" and c.params == {"i": 1}


def test_config_hash_is_frozen():
    # Pinned literal: SimpleRunConfig's serialization is a contract, not an impl
    # detail -- these bytes name run directories. A change here is ledger-breaking.
    c = SimpleRunConfig(name="demo", dtype="float32", params={"R": 2.0})
    assert c.to_json() == '{"dtype": "float32", "name": "demo", "params": {"R": 2.0}}'
    assert c.config_hash() == c.config_hash()             # deterministic
    assert c.run_name() == f"demo_{c.config_hash()}"
    assert len(c.config_hash(8)) == 8


def test_config_hash_is_order_insensitive():
    """A worker rebuilding params from JSON gets whatever key order the serializer
    emitted; an order-sensitive hash would fork identity between driver and box."""
    a = SimpleRunConfig(params={"z": 1, "a": 2})
    b = SimpleRunConfig(params={"a": 2, "z": 1})
    assert a.config_hash() == b.config_hash()


def test_from_json_round_trips_identity():
    c = SimpleRunConfig(name="x", dtype="float64", params={"a": {"b": [1, 2]}})
    back = SimpleRunConfig.from_json(c.to_json())
    assert back == c and back.config_hash() == c.config_hash()


def test_from_json_is_lossy_in_type_but_not_hash():
    """The documented contract edge, both directions. Hash stays stable (identity
    and idempotent-skip keep working) while the Python type is NOT preserved -- so
    a RunFn must accept the JSON shadow or keep params JSON-native."""
    @dataclasses.dataclass(frozen=True)
    class Rates:
        point: float = 0.01

    c = SimpleRunConfig(params={"rates": Rates(), "rng": (1, 2)})
    back = SimpleRunConfig.from_json(c.to_json())
    # hash unchanged...
    assert back.config_hash() == c.config_hash()
    # ...but the types collapsed to their JSON shadows
    assert isinstance(back.params["rates"], dict)         # dataclass -> dict
    assert back.params["rates"] == {"point": 0.01}
    assert isinstance(back.params["rng"], list)           # tuple -> list
    assert back.params["rng"] == [1, 2]


def test_checkpoint_round_trip_is_full_state(tmp_path):
    cfg = SimpleRunConfig(name="ck", params={"R": 2.0})
    state = {"z": np.arange(6, dtype=np.float64).reshape(2, 3),
             "v": np.ones(4, dtype=np.float64)}
    p = tmp_path / "ck.npz"
    save_checkpoint(p, state, cfg, step=7)
    got, back, step = load_checkpoint(p, config_class=SimpleRunConfig)
    assert step == 7
    assert back == cfg                                    # config rebuilt via config_class
    assert np.array_equal(np.asarray(got["z"]), state["z"])
    assert np.array_equal(np.asarray(got["v"]), state["v"])


def test_load_checkpoint_default_config_class(tmp_path):
    cfg = SimpleRunConfig(name="d", params={})
    p = tmp_path / "d.npz"
    save_checkpoint(p, {"x": np.zeros(2)}, cfg, step=1)
    _state, back, _step = load_checkpoint(p)              # default SimpleRunConfig
    assert isinstance(back, SimpleRunConfig) and back == cfg


def test_run_dir_writes_config_and_manifest(tmp_path):
    cfg = SimpleRunConfig(name="rd", params={"i": 3})
    d = run_dir(tmp_path, cfg)
    assert d.name == cfg.run_name()
    assert (d / "config.json").read_text().strip() == cfg.to_json()
    import json
    rows = [json.loads(x) for x in
            (tmp_path / "MANIFEST.jsonl").read_text().splitlines() if x.strip()]
    assert rows == [{"run": cfg.run_name(), "config": json.loads(cfg.to_json())}]
    # idempotent: a second call doesn't duplicate the manifest row
    run_dir(tmp_path, cfg)
    rows2 = [x for x in (tmp_path / "MANIFEST.jsonl").read_text().splitlines() if x.strip()]
    assert len(rows2) == 1
