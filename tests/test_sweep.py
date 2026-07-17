"""sweep.legs -- the (arm x replicate x grid) cross-product expansion.

The properties that matter: every axis lands on every leg (so aggregation can
group by them), seed_fn is CALLED not derived (the caller owns seed provenance),
the enumeration is deterministic (a re-launch resumes identically), and grid cells
are copied (a caller reusing a cell dict can't retroactively change a leg).
"""

from run_farm.sweep import legs


def test_cross_product_is_complete():
    out = list(legs(["ctrl", "treat"], [0, 1],
                    [{"eps": 0.3}, {"eps": 0.5}],
                    seed_fn=lambda a, r, c: 0))
    assert len(out) == 2 * 2 * 2                          # arms x reps x cells
    # every combination present exactly once
    combos = {(l["arm"], l["replicate"], l["cfg"]["eps"]) for l in out}
    assert len(combos) == 8


def test_axes_present_on_every_leg():
    out = list(legs(["a"], [7], [{"k": 1}], seed_fn=lambda a, r, c: 99))
    leg = out[0]
    assert leg["arm"] == "a" and leg["replicate"] == 7
    assert leg["cfg"] == {"k": 1} and leg["seed"] == 99
    assert "rid" in leg


def test_seed_fn_is_called_not_derived():
    """run-farm must NOT invent seeds -- a consumer's draw order can be load-bearing
    for reproducing published results. Prove the supplied fn is what's used."""
    calls = []

    def seed_fn(arm, rep, cell):
        calls.append((arm, rep, dict(cell)))
        return rep * 1000 + int(cell["eps"] * 10)

    out = list(legs(["x"], [0, 1], [{"eps": 0.3}], seed_fn=seed_fn))
    assert [l["seed"] for l in out] == [3, 1003]
    assert len(calls) == 2                                # called once per leg


def test_enumeration_is_deterministic():
    mk = lambda: list(legs(["a", "b"], [0, 1], [{"g": 1}, {"g": 2}],
                           seed_fn=lambda a, r, c: 0))
    assert [l["rid"] for l in mk()] == [l["rid"] for l in mk()]


def test_empty_grid_is_pure_arm_x_replicate():
    out = list(legs(["a", "b"], [0], seed_fn=lambda a, r, c: 0))
    assert len(out) == 2
    assert all(l["cfg"] == {} for l in out)


def test_grid_cell_is_copied():
    """A caller reusing/mutating a cell dict after expansion must not change a
    leg's identity."""
    cell = {"eps": 0.3}
    out = list(legs(["a"], [0], [cell], seed_fn=lambda a, r, c: 0))
    cell["eps"] = 0.9
    assert out[0]["cfg"] == {"eps": 0.3}


def test_default_rid_is_stable_and_legible():
    out = list(legs(["ctrl"], [2], [{"eps": 0.3, "k": 5}], seed_fn=lambda a, r, c: 0))
    assert out[0]["rid"] == "ctrl_r2_eps0.3_k5"           # sorted keys, composite


def test_custom_rid_fn():
    out = list(legs(["ctrl"], [0], [{"i": 1}], seed_fn=lambda a, r, c: 0,
                    rid_fn=lambda a, r, c: f"custom-{a}-{c['i']}"))
    assert out[0]["rid"] == "custom-ctrl-1"
