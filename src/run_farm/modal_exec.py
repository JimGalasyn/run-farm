"""ModalExecutor: serverless campaign fan-out -- the Executor (D) seam for Modal.

Modal is NOT a `Provider` (F): there is no host to rent and no meter to leak.
You hand it a containerized function and it provisions, scales, per-second
bills, retries, and tears down. So Modal plugs in here, at the executor seam,
and the lifecycle worries the Provider contract carries simply don't exist.

The fan-out is `Function.map` over the configs; each call runs the shared
`run_one` unit (register/skip/resume/run/finish, identical to the in-process
driver) against a **Modal Volume** mounted as the registry root, so checkpoints,
event streams, and triggered captures persist and `is_complete`/resume work
across runs. Small result records ride the `.map` return values; the heavy
artifacts stay on the Volume (fetch with `modal volume get`, or mount it).

`modal` is an optional dependency -- importing this module requires it, but
`import run_farm` does not.

    from run_farm.modal_exec import ModalExecutor
    ex = ModalExecutor("mypkg.runfns:relax_then_measure", gpu="A10G")
    results = ex.run(configs)            # configs: Iterable[RunConfig]
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

import modal

from run_farm.remote import RunFnRef, run_one
from run_farm.protocols import RunConfig

_MOUNT = "/campaign"
DEFAULT_VOLUME = "run-farm-campaign"


def default_image(*packages: str, python_version: str | None = None,
                  ref: str = "main") -> "modal.Image":
    """A debian image with git + jax (CUDA 12) + run-farm + YOUR engine.

    `packages` are extra pip specs -- and for a real campaign at least one of them
    must supply `run_fn_ref`, because run-farm ships no physics. Anything pip
    understands works::

        default_image("jax-solitons")                        # from PyPI
        default_image("git+https://github.com/me/eng@my-branch")   # unmerged work

    A VCS spec is worth calling out: the ref you pin MUST contain the code the
    worker runs, so while a feature branch is unmerged, install from it; a release
    pins a tag. (`debian_slim` has no git, which a VCS install needs -- hence
    `apt_install("git")`.) Pass a full `image` to ModalExecutor to build it
    yourself instead.

    `ref` pins run-farm's own git ref. It exists for the same reason: a worker
    running `run_farm.remote.run_one` from a stale image against a driver on newer
    code is a version-skew bug that looks like a broken host.
    """
    py = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        modal.Image.debian_slim(python_version=py)
        .apt_install("git")
        .pip_install("jax[cuda12]",
                     f"git+https://github.com/JimGalasyn/run-farm@{ref}",
                     *packages)
    )


def _modal_worker(config_json: str, run_fn_ref: str, work_dir: str,
                  volume_name: str) -> dict:
    """The function Modal runs in-container (referenced by name, not pickled, so
    the image's own engine provides it). Reload the Volume to see prior
    checkpoints (resume/skip), run the shared unit, commit to persist artifacts.
    """
    vol = modal.Volume.from_name(volume_name)
    try:
        vol.reload()
    except Exception:
        pass
    out = run_one(config_json, run_fn_ref, work_dir)
    try:
        vol.commit()                # persist checkpoints/events/triggers
    except Exception as e:          # results still return; durability best-effort
        out["volume_commit"] = f"failed: {e}"
    return out


class ModalExecutor:
    """Run a campaign across Modal GPUs via `Function.map`.

    `run_fn_ref` is the ``'module:function'`` RunFn the workers import (the one
    physics injection; see `run_farm.remote`). `image` must have run-farm + the engine
    (and the package holding `run_fn_ref`) installed; defaults to `default_image`.
    """

    name = "modal"

    def __init__(self, run_fn_ref: RunFnRef, *, image: "modal.Image | None" = None,
                 gpu: str = "A10G", app_name: str = "run-farm-campaign",
                 volume_name: str = DEFAULT_VOLUME, work_subdir: str = "runs",
                 timeout: int = 3600, retries: int = 2):
        self.run_fn_ref = run_fn_ref
        self.gpu = gpu
        self.volume_name = volume_name
        self.work_dir = f"{_MOUNT}/{work_subdir}"
        self.volume = modal.Volume.from_name(volume_name, create_if_missing=True)
        self.app = modal.App(app_name)
        # Module-level worker referenced by name (not cloudpickled): the image's
        # own engine provides it, so there's no module-availability or
        # Python-version-match fragility.
        self._worker = self.app.function(
            image=image or default_image(), gpu=gpu,
            volumes={_MOUNT: self.volume}, timeout=timeout,
            retries=retries)(_modal_worker)

    def run(self, configs: Iterable[RunConfig], *, admission=None) -> list[dict]:
        """Fan `configs` out across Modal GPUs; return the small result records.

        `admission` is accepted for parity with the in-process executor but is a
        no-op: Modal manages reliable hosts, so the P9 probe-or-bail (E) -- whose
        whole point is flaky marketplace hosts -- does not apply here.
        """
        cfg_jsons = [c.to_json() for c in configs]
        if not cfg_jsons:
            return []
        with self.app.run():
            return list(self._worker.map(
                cfg_jsons, kwargs={"run_fn_ref": self.run_fn_ref,
                                   "work_dir": self.work_dir,
                                   "volume_name": self.volume_name}))
