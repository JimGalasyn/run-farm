"""A second reference `Provider` (F): a direct RunPod broker (stdlib only).

`RunPodProvider` is the rule-of-three test of the `Provider` Protocol -- a
second cloud behind the same `offers` + teardown-verifying `rent` seam, proving the
abstraction wasn't shaped to one marketplace. It also documents, in code, the
two ways RunPod differs from Vast that the Protocol had to absorb:

  1. **Offers are GPU *types*, not hosts.** RunPod's REST API
     (`rest.runpod.io/v1`) is pod-centric and has no host catalog; the GPU-type
     list + pricing lives in the GraphQL API (`gpuTypes`). So `offers()` queries
     GraphQL and returns one synthetic `Offer` per available (type, cloud-tier),
     `id` = the gpuTypeId (e.g. "NVIDIA GeForce RTX 4090"); `rent()` then asks
     RunPod to *place* a pod of that type rather than picking a named machine.
     Per-host metrics Vast exposes at selection (reliability, inet) aren't in the
     catalog -- they come out as NaN ("unknown"), honestly.

  2. **Admission is split, not all-at-selection.** Vast filters everything in
     `offers()`; RunPod resolves CUDA floor and bandwidth at *pod create*
     (`allowedCudaVersions`, `minDownloadMbps`). So those gates are provider
     config applied in `rent()`, not `offers()` filters. `offers()` honors
     gpu_name / max_dph / tier; the create-time gates default from the most
     recent `offers(spec)` (single-spec-per-provider, as a campaign uses it) or
     from constructor overrides.

Cost safety is the same structural invariant as `VastProvider`: `rent()` ALWAYS
terminates on exit and verifies the pod is gone, raising on a confirmed leak.

API key: ``$RUNPOD_API_KEY``, else ``~/.runpod_api_key``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from run_farm.protocols import (
    HostProbeFailed,
    HostSpec,
    LaunchSpec,
    LeakRisk,
    Offer,
    RentedHost,
)

REST = "https://rest.runpod.io/v1"
GQL = "https://api.runpod.io/graphql"
_KEY_PATHS = ("~/.runpod_api_key",)

# CUDA versions RunPod accepts in `allowedCudaVersions`; we send those >= the
# requested floor so a host whose driver is too old for the image is never
# placed (the P10 image-floor gate, RunPod's native form).
_CUDA_LADDER = ("11.8", "12.0", "12.1", "12.2", "12.3", "12.4", "12.5",
                "12.6", "12.7", "12.8", "13.0")

# Pod statuses that mean "this host will never come up" -> fail over (P9).
_DEAD_STATUSES = ("TERMINATED", "FAILED", "EXITED", "DEAD")


@dataclasses.dataclass(frozen=True)
class Pod:
    """A live pod as the reaper/control-plane sees it -- the RunPod analogue of
    `vast.Instance` (`id`/`status`/`dph`/`raw`), so `reap` is provider-agnostic.
    `raw['label']` is normalized from the pod name (RunPod's attribution field)
    so `reap --label` reads the same key on either cloud."""

    id: str
    status: str
    dph: float
    raw: dict


class RunPodError(RuntimeError):
    """A RunPod API call failed (status quoted, key never echoed).

    `code` carries the HTTP status on an HTTP error (None for transport/GraphQL),
    so teardown can treat a 404/410 (pod already gone) as success instead of
    retrying a nonexistent pod and raising a spurious leak (CM review #48)."""

    code: int | None = None


def _read_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("RUNPOD_API_KEY"):
        return os.environ["RUNPOD_API_KEY"].strip()
    for p in _KEY_PATHS:
        fp = Path(p).expanduser()
        if fp.exists():
            return fp.read_text().strip()
    raise RuntimeError(
        "no RunPod API key: set $RUNPOD_API_KEY or write ~/.runpod_api_key")


# RunPod's API is behind Cloudflare, which 403s (error 1010) the default
# `Python-urllib/x.y` User-Agent. Any explicit UA clears it -- a footgun Vast
# (no Cloudflare UA filter) didn't have.
_UA = "run-farm-runpod-provider"


def _req(method: str, url: str, key: str, payload=None, timeout: float = 30):  # pragma: no cover
    """One JSON HTTP call; raises RunPodError on a non-2xx (body quoted).

    Live HTTP to RunPod; unit tests monkeypatch this, so its body isn't covered."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json",
               "User-Agent": _UA}
    if data is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode(errors="replace")
        err = RunPodError(f"{method} {url.split('?')[0]} -> HTTP {e.code}: {detail}")
        err.code = e.code                                # structured status for callers
        raise err from None


class RunPodProvider:
    """The RunPod reference `Provider` (F): GraphQL catalog + REST pod lifecycle.

    `cloud_type` picks the tier ("COMMUNITY" cheap/spot-like, "SECURE"
    datacenter); `interruptible` requests spot pods. `min_cuda` / `min_download_mbps`
    are RunPod's create-time admission gates (defaulted from the HostSpec at
    `offers()` time if not set here). `ledger` is any object with a
    `record(event, **fields)` method (the VastLedger works as-is).
    """

    name = "runpod"

    def __init__(self, api_key: str | None = None, ledger=None, *,
                 cloud_type: str = "COMMUNITY", interruptible: bool = False,
                 min_cuda: float | None = None,
                 min_download_mbps: float | None = None,
                 data_center_ids: list[str] | None = None,
                 volume_gb: int = 0):
        self.key = _read_key(api_key)
        self.ledger = ledger
        self.cloud_type = cloud_type.upper()
        self.interruptible = interruptible
        self.data_center_ids = data_center_ids
        self.volume_gb = volume_gb
        # Create-time admission: explicit constructor args win and are pinned;
        # otherwise each offers() call refreshes them from the HostSpec it sees,
        # so reusing the provider with a different spec sends the current gates.
        self._min_cuda = min_cuda
        self._min_inet = min_download_mbps
        self._cuda_pinned = min_cuda is not None
        self._inet_pinned = min_download_mbps is not None

    def _log(self, event: str, **fields) -> None:
        if self.ledger is not None:
            self.ledger.record(event, **fields)

    def _price_for_tier(self, g: dict) -> float | None:
        return g.get("communityPrice") if self.cloud_type == "COMMUNITY" \
            else g.get("securePrice")

    def _tier_available(self, g: dict) -> bool:
        return bool(g.get("communityCloud")) if self.cloud_type == "COMMUNITY" \
            else bool(g.get("secureCloud"))

    # -- discovery (free; no spend) ------------------------------------------
    def offers(self, spec: HostSpec) -> list[Offer]:
        """Available GPU *types* meeting `spec`, cheapest first (the F discovery
        half, RunPod-shaped). See the module docstring: RunPod has no host
        catalog, so an Offer is a (type, cloud-tier) with `id` = the gpuTypeId.
        Per-host reliability/inet aren't in the catalog -> NaN (unknown); they
        are enforced at rent via RunPod's create-time admission instead."""
        # Refresh the create-time gates from this spec unless the constructor
        # pinned them (so a reused provider tracks the most recent spec).
        if not self._cuda_pinned:
            self._min_cuda = spec.min_cuda
        if not self._inet_pinned:
            self._min_inet = spec.min_inet_mbps
        want = spec.gpu_name.replace("_", " ").lower()
        q = ("query { gpuTypes { id displayName memoryInGb "
             "secureCloud communityCloud securePrice communityPrice } }")
        resp = _req("POST", GQL, self.key, {"query": q})
        if resp.get("errors"):
            raise RunPodError(f"gpuTypes query: {json.dumps(resp['errors'])[:200]}")
        out = []
        for g in resp.get("data", {}).get("gpuTypes", []):
            name = g.get("displayName") or g.get("id") or ""
            if want not in name.lower() and want not in str(g.get("id", "")).lower():
                continue
            if not self._tier_available(g):
                continue
            price = self._price_for_tier(g)
            if price is None or price > spec.max_dph:
                continue
            out.append(Offer(
                id=str(g["id"]), dph=float(price), gpu_name=name,
                num_gpus=spec.num_gpus,
                reliability=float("nan"), inet_down_mbps=float("nan"),
                cuda_max=float("nan"), geolocation=self.cloud_type,
                provider=self.name))
        return sorted(out, key=lambda o: o.dph)

    def cheapest_offer(self, spec: HostSpec) -> Offer | None:
        offs = self.offers(spec)
        return offs[0] if offs else None

    # -- lifecycle -----------------------------------------------------------
    def _allowed_cuda(self) -> list[str]:
        floor = self._min_cuda or 0.0
        return [v for v in _CUDA_LADDER if float(v) >= floor]

    def create(self, offer: Offer, launch: LaunchSpec, *, attempts: int = 6) -> str:
        """Create (place) a pod of `offer`'s GPU type; returns the pod id.

        RunPod picks a machine at create time and can 500 with 'does not have the
        resources ... try a different machine' when the chosen host is full. That
        is transient capacity, not a bad request, and each retry may land on a
        different machine -- so retry it (unlike RunPod's per-type catalog, which
        gives the executor nothing to fail over to). Other errors propagate."""
        body = {
            "name": launch.label,                        # attribution -> reap --label
            "imageName": launch.image,
            "gpuTypeIds": [offer.id],
            "gpuCount": offer.num_gpus,
            "cloudType": self.cloud_type,
            "computeType": "GPU",
            "containerDiskInGb": int(launch.disk_gb),
            "volumeInGb": int(self.volume_gb),
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "interruptible": self.interruptible,
            "dockerStartCmd": ["bash", "-c", launch.onstart],
            "allowedCudaVersions": self._allowed_cuda(),
        }
        if self._min_inet:
            body["minDownloadMbps"] = self._min_inet
        if self.data_center_ids:
            body["dataCenterIds"] = self.data_center_ids
        last = ""
        for _ in range(attempts):
            try:
                res = _req("POST", f"{REST}/pods", self.key, body)
                pid = res.get("id")
                if pid:
                    return str(pid)
                last = f"no pod id: {str(res)[:160]}"
            except RunPodError as e:
                last = str(e)
                low = last.lower()
                if "resource" not in low and "try a different machine" not in low:
                    raise                        # a real error, not capacity
            time.sleep(3)
        raise RunPodError(
            f"create failed after {attempts} attempts (capacity?): {last[:200]}")

    def status(self, pod_id: str) -> dict:
        return _req("GET", f"{REST}/pods/{pod_id}", self.key)

    def terminate(self, pod_id: str) -> None:
        _req("DELETE", f"{REST}/pods/{pod_id}", self.key)

    # -- reap / control-plane contract (provider-agnostic) -------------------
    def destroy(self, pod_id) -> None:
        """Alias for `terminate`, the name `reap` and the multi-cloud control
        plane expect (matching `VastProvider.destroy`). Idempotent."""
        self.terminate(str(pod_id))

    def list_instances(self) -> list[Pod]:
        """All of the account's pods, as `Pod`s -- the reaper's source of truth.
        `costPerHr` field names vary; fall back to 0 (only the spend report uses
        it). `raw['label']` is normalized from the pod name for `reap --label`."""
        pods = _req("GET", f"{REST}/pods", self.key)
        return [Pod(id=str(p.get("id")),
                    status=str(p.get("desiredStatus") or "?"),
                    dph=float(p.get("costPerHr") or p.get("adjustedCostPerHr") or 0),
                    raw={**p, "label": p.get("name")})
                for p in pods]

    def dead_reason(self, pod_id) -> str | None:
        """A reason if the pod has visibly failed (a `_DEAD_STATUSES` desired
        status), else None -- the FleetExecutor fast-fail hook, at parity with
        `VastProvider.dead_reason`."""
        try:
            st = str(self.status(str(pod_id)).get("desiredStatus") or "").upper()
        except Exception:
            return None
        return f"pod {pod_id} -> {st}" if st in _DEAD_STATUSES else None

    def _present(self, pod_id: str) -> bool:
        """True if the pod still exists -- the independent teardown verify."""
        try:
            pods = _req("GET", f"{REST}/pods", self.key)
        except RunPodError:
            raise
        return any(str(p.get("id")) == pod_id for p in pods)

    @staticmethod
    def _ssh_coords(pod: dict) -> tuple[str, int]:
        """Public IP + the host port mapped to container 22, or ('', 0) if not
        yet assigned. `portMappings` is {container_port: public_port}."""
        ip = str(pod.get("publicIp") or "")
        pm = pod.get("portMappings") or {}
        port = pm.get("22") or pm.get(22) or 0
        return ip, int(port or 0)

    def wait_running(self, pod_id: str, *, timeout_s: float = 600,
                     poll_s: float = 10) -> dict:
        """Poll until the pod is RUNNING with SSH reachable; raise
        HostProbeFailed on a dead status (P9 failover signal) or TimeoutError."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pod = self.status(pod_id)
            st = str(pod.get("desiredStatus") or "").upper()
            if st in _DEAD_STATUSES:
                raise HostProbeFailed(f"pod {pod_id} -> {st}")
            if st == "RUNNING":
                ip, port = self._ssh_coords(pod)
                if ip and port:
                    return pod
            time.sleep(poll_s)
        raise TimeoutError(f"pod {pod_id} not SSH-ready in {timeout_s}s")

    # -- the safety primitive (the Provider F invariant) ---------------------
    @contextlib.contextmanager
    def rent(self, offer: Offer, launch: LaunchSpec, *, timeout_s: float = 600):
        """Place a pod of `offer`'s type, wait until SSH-ready, yield a
        `RentedHost`, and terminate+verify on every exit it can intercept -- the teardown-verify
        contract every Provider owes. Raises `HostProbeFailed` if the pod never
        comes up usable, so the executor can fail over to the next offer."""
        t0 = time.monotonic()
        pod_id = self.create(offer, launch)
        self._log("rented", provider=self.name, offer_id=offer.id,
                  instance_id=pod_id, gpu=offer.gpu_name, dph=offer.dph,
                  geo=offer.geolocation)
        outcome, reason = "ok", ""
        try:
            pod = self.wait_running(pod_id, timeout_s=timeout_s)
            ip, port = self._ssh_coords(pod)
            self._log("running", provider=self.name, offer_id=offer.id,
                      instance_id=pod_id,
                      provision_s=round(time.monotonic() - t0, 1))
            yield RentedHost(id=pod_id, ssh_host=ip, ssh_port=port,
                             offer=offer, raw=pod)
        except HostProbeFailed as e:
            outcome, reason = "host_failed", str(e)
            raise
        except TimeoutError as e:
            outcome, reason = "timeout", str(e)
            raise
        except BaseException as e:                # propagate, but record the cause
            outcome, reason = type(e).__name__, str(e)[:200]
            raise
        finally:
            billed_s = time.monotonic() - t0
            destroyed = False
            for _ in range(5):
                try:
                    self.terminate(pod_id)
                    destroyed = True
                    break
                except RunPodError as e:
                    if e.code in (404, 410):            # already gone = desired state
                        destroyed = True
                        break
                    time.sleep(2)
                except Exception:
                    time.sleep(2)
            try:
                verify = "present" if self._present(pod_id) else "gone"
            except Exception as e:
                verify = f"unverified: {e}"
            self._log("destroyed", provider=self.name, offer_id=offer.id,
                      instance_id=pod_id, outcome=outcome, reason=reason[:200],
                      billed_s=round(billed_s, 1),
                      est_cost_usd=round(offer.dph * billed_s / 3600, 4),
                      destroyed=destroyed, verify=verify)
            if not destroyed or verify == "present":
                raise LeakRisk(                        # structured (not stringly-typed)
                    f"LEAK RISK: pod {pod_id} not confirmed terminated "
                    f"(destroyed={destroyed}, verify={verify}) -- "
                    f"terminate it at https://console.runpod.io/pods")
            if verify.startswith("unverified"):              # pragma: no cover
                print(f"  ~ pod {pod_id}: terminated but teardown UNVERIFIED "
                      f"({verify[:80]}) -- confirm it is gone.")
