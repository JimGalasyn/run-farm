"""The reference `Provider` (F): a direct Vast.ai HTTP broker (stdlib only).

`VastProvider` implements the campaign `Provider` Protocol -- `offers(HostSpec)`
and a leak-proof `rent(Offer, LaunchSpec)` -- so the campaign can rent Vast GPUs
behind the same seam any other cloud plugs into. It is also the worked example a
new adapter (RunPod, TensorDock, EC2 spot) copies.

Why a direct client and not an SDK (P10): SkyPilot's Vast provider AND the
official `vastai` SDK both break against Vast's current API -- the bare
collection endpoint ``GET /api/v0/instances/`` returns HTTP 410 Gone, and both
route instance-listing there. Probing showed the rest of v0 is alive
(``/instances/{id}/`` -> 200), and v1 serves the list. So this talks straight to
the endpoints that work -- **v1 for listing, v0 sub-resources for
create/destroy/show/logs** -- with no third-party dependency (urllib only),
which also means nothing to break when the SDK lags the API again.

Cost safety is structural and IS the Provider contract: ``rent()`` ALWAYS
destroys on exit (success, exception, or Ctrl-C) and then verifies via the
independent v1 list endpoint that the instance is actually gone. A leaked GPU
bills by the second.

API key: ``$VAST_API_KEY``, else ``~/.config/vastai/vast_api_key``, else
``~/.vast_api_key`` (the first that exists; see ``_KEY_PATHS``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from run_farm.ledger import RentalLedger
from run_farm.protocols import (
    HostProbeFailed,
    HostSpec,
    LaunchSpec,
    LeakRisk,
    Offer,
    RentedHost,
    RentUnavailable,
)

# RentalLedger was lifted to run_farm.ledger (it never was Vast-specific -- RunPod
# uses it too). Alias kept so `run_farm.vast.VastLedger` still resolves.
VastLedger = RentalLedger

V0 = "https://console.vast.ai/api/v0"
V1 = "https://console.vast.ai/api/v1"
_KEY_PATHS = ("~/.config/vastai/vast_api_key", "~/.vast_api_key")

# Host-failure signatures seen in an instance's status_msg DURING provisioning.
# A host can advertise high reliability + bandwidth and still be unable to
# resolve DNS / pull an image -- the metadata lies (P9). Catching these lets
# wait_running bail in seconds instead of waiting out the full timeout.
_BAD_HOST_SIGNS = (
    "failed to resolve", "i/o timeout", "failed to authorize", "no such host",
    "connection refused", "error response from daemon", "manifest unknown",
    "unauthorized", "temporary failure in name resolution", "no space left",
)


class VastError(RuntimeError):
    """A Vast API call failed (status quoted, key never echoed).

    `code` carries the HTTP status when the failure was an HTTP error (None for a
    transport/DNS failure), so a caller can distinguish a terminal auth/config
    error (401/403) from a recoverable one without parsing the message."""

    code: int | None = None


# -- transient-fault retry (issue #23) ---------------------------------------
# A single saturated local resolver (an OneDrive sync, a DNS storm) throws
# EAI_AGAIN and made a whole 16-leg farm treat every rent as TERMINALLY failed
# -- a 1-second retry would have made the storm invisible. So every Vast REST
# call self-heals on a *transient* fault before giving up. The HTTP codes worth
# retrying are the "try again" family; a 4xx (bad request / auth / 404) is
# terminal and raises at once.
_RETRY_HTTP = frozenset({408, 425, 429, 500, 502, 503, 504})
_MAX_TRIES = 5
_BACKOFF_BASE = 0.5              # 0.5, 1, 2, 4 s ... (capped)
_BACKOFF_CAP = 8.0
# Only a TEMPORARY resolver failure (EAI_AGAIN -- the "temporary failure in name
# resolution" a saturated resolver throws) is worth retrying. A name that
# genuinely doesn't resolve (EAI_NONAME / EAI_FAIL) is terminal: retrying it just
# burns backoff and hides a misconfiguration. (EAI_AGAIN is POSIX-standard; the
# getattr keeps this importable on a platform that somehow lacks it.)
_TRANSIENT_GAIERROR = frozenset(
    getattr(socket, n) for n in ("EAI_AGAIN",) if hasattr(socket, n))


def _transient(exc: BaseException, *, idempotent: bool) -> bool:
    """True if `exc` is a transient transport fault worth retrying.

    The `idempotent` distinction is a cost-safety invariant, not a nicety: a
    TEMPORARY DNS failure (`socket.gaierror` with EAI_AGAIN, the "temporary
    failure in name resolution" a saturated resolver throws) happens BEFORE the
    request reaches Vast, so the server never acted on it and retrying is always
    safe -- even a non-idempotent `create`. Only EAI_AGAIN qualifies; a name that
    genuinely doesn't resolve (EAI_NONAME/EAI_FAIL) is terminal. A post-connection
    fault (reset, read timeout, 5xx) might mean the request WAS received and acted
    on, so it is retried only for idempotent calls; retrying a `create` that
    actually succeeded would rent a second GPU that bills by the second (the very
    leak the Provider contract exists to prevent)."""
    if isinstance(exc, urllib.error.HTTPError):
        return idempotent and exc.code in _RETRY_HTTP
    if isinstance(exc, urllib.error.URLError):
        if isinstance(exc.reason, socket.gaierror):
            return exc.reason.errno in _TRANSIENT_GAIERROR   # EAI_AGAIN: pre-send
        return idempotent                    # connect/read transport fault
    if isinstance(exc, (socket.timeout, TimeoutError, ConnectionError)):
        return idempotent
    return False


# `HostProbeFailed` is the campaign-wide failover signal (campaign.protocols),
# imported above so every Provider raises the SAME exception the executor
# fails over on -- not a Vast-private subclass.


def _read_key(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if os.environ.get("VAST_API_KEY"):
        return os.environ["VAST_API_KEY"].strip()
    for p in _KEY_PATHS:
        fp = Path(p).expanduser()
        if fp.exists():
            return fp.read_text().strip()
    raise RuntimeError(
        "no Vast API key: set $VAST_API_KEY or write ~/.config/vastai/vast_api_key")


def _req(method: str, url: str, key: str, payload=None, timeout: float = 30,
         *, idempotent: bool = True, tries: int = _MAX_TRIES):
    """One Vast REST call, self-healing on transient transport faults (#23).

    Retries `tries` times with exponential backoff on a transient fault (DNS
    EAI_AGAIN, connection reset, read timeout, 5xx); a terminal 4xx raises at
    once. `idempotent=False` (used only by `create`) narrows the retry to
    pre-send DNS failures so a half-completed create can't double-rent. On
    exhaustion the last fault is wrapped in `VastError` -- a raw `URLError` no
    longer escapes to callers, so every network failure looks the same to the
    executor's failover path."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    endpoint = url.split("?")[0]
    tries = max(1, tries)                       # at least one attempt; keeps the
    for attempt in range(tries):                # post-loop raise genuinely unreachable
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            if not (_transient(e, idempotent=idempotent) and attempt < tries - 1):
                detail = e.read()[:200].decode(errors="replace")
                err = VastError(f"{method} {endpoint} -> HTTP {e.code}: {detail}")
                err.code = e.code                # let callers classify (auth vs race)
                e.close()
                raise err
            e.close()                           # retrying: release the response fd
        except (urllib.error.URLError, socket.timeout, TimeoutError,
                ConnectionError) as e:
            if not (_transient(e, idempotent=idempotent) and attempt < tries - 1):
                reason = getattr(e, "reason", e)
                raise VastError(
                    f"{method} {endpoint} -> {type(e).__name__}: {reason}") from e
        time.sleep(min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt)))
    raise VastError(  # pragma: no cover  (tries>=1, so the loop returns or raises)
        f"{method} {endpoint} failed after {tries} tries")


@dataclasses.dataclass(frozen=True)
class Instance:
    id: int
    status: str
    dph: float
    raw: dict




class VastProvider:
    """The reference `Provider` (F): a direct Vast.ai HTTP broker.

    Implements the campaign `Provider` Protocol (`offers` + leak-proof `rent`)
    so Vast plugs into the same seam as any other cloud, and is the worked
    example a new adapter copies. The lower-level lifecycle methods
    (`create`/`status`/`wait_running`/`logs`/`destroy`) are public for direct
    operational use but the contract surface is just `offers` and `rent`."""

    name = "vast"

    def __init__(self, api_key: str | None = None,
                 ledger: "VastLedger | None" = None):
        self.key = _read_key(api_key)
        self.ledger = ledger

    def _log(self, event: str, **fields) -> None:
        if self.ledger is not None:
            self.ledger.record(event, **fields)

    # -- discovery (free; no spend) ------------------------------------------
    def offers(self, spec: HostSpec) -> list[Offer]:
        """Rentable offers meeting `spec`, cheapest first (the F discovery half).

        The filters are the P9 admission spirit at SELECTION time -- but the
        live failure (a 0.99-reliability host that can't resolve DNS) shows
        selection metadata is necessary, not sufficient. `rent` must still
        probe-and-bail per host (see wait_running) and the executor fails over
        to the next. `cuda_max_good >= spec.min_cuda` is the P10 gate: a host
        whose driver is older than the launch image fails container-create, so
        it is never surfaced."""
        q = {
            "verified": {"eq": True}, "rentable": {"eq": True},
            "rented": {"eq": False},
            "gpu_name": {"eq": spec.gpu_name.replace("_", " ")},
            "num_gpus": {"eq": spec.num_gpus},
            "reliability2": {"gte": spec.min_reliability},
            "inet_down": {"gte": spec.min_inet_mbps},
            "cuda_max_good": {"gte": spec.min_cuda},
            "order": [["dph_total", "asc"]], "type": "on-demand",
            "limit": 64, "allocated_storage": 5.0,
        }
        frac = getattr(spec, "min_gpu_frac", 0.0)       # dedicated-machine gate (anti-contention)
        if frac > 0:
            q["gpu_frac"] = {"gte": frac}
        raw = _req("POST", f"{V0}/bundles/", self.key, q).get("offers", [])
        return [
            Offer(id=str(o["id"]), dph=float(o["dph_total"]),
                  gpu_name=o["gpu_name"], num_gpus=int(o["num_gpus"]),
                  reliability=float(o.get("reliability2", 0)),
                  inet_down_mbps=float(o.get("inet_down", 0)),
                  cuda_max=float(o.get("cuda_max_good", 0)),
                  geolocation=o.get("geolocation", ""), provider=self.name)
            for o in raw if o.get("dph_total", 1e9) <= spec.max_dph]

    def cheapest_offer(self, spec: HostSpec) -> Offer | None:
        offs = self.offers(spec)
        return offs[0] if offs else None

    # -- lifecycle -----------------------------------------------------------
    def create(self, offer_id: int | str, *, image: str, onstart_cmd: str,
               disk: float = 40.0, label: str = "run-farm",
               runtype: str = "ssh") -> int:
        """Create an instance from an offer; returns the instance id."""
        try:
            ask_id = int(offer_id)
        except (TypeError, ValueError):
            raise VastError(
                f"create: {offer_id!r} is not a Vast ask id -- an Offer from "
                f"another Provider cannot be rented through VastProvider")
        blob = {"client_id": "me", "image": image, "env": {}, "disk": disk,
                "label": label, "onstart": onstart_cmd, "runtype": runtype}
        # idempotent=False: a create that the server received but whose response
        # was lost would, on retry, rent a SECOND GPU. So only pre-send DNS
        # failures retry here (see `_transient`); any post-connect fault fails.
        res = _req("PUT", f"{V0}/asks/{ask_id}/", self.key, blob, idempotent=False)
        new_id = res.get("new_contract") or res.get("id")
        if not new_id:
            raise VastError(f"create returned no instance id: {res}")
        return int(new_id)

    def status(self, instance_id: int) -> Instance:
        d = _req("GET", f"{V0}/instances/{instance_id}/?owner=me", self.key)
        inst = d.get("instances", d) or {}
        return Instance(id=instance_id, status=inst.get("actual_status", "?"),
                        dph=float(inst.get("dph_total", 0) or 0), raw=inst)

    def list_instances(self) -> list[Instance]:
        """All of the account's instances, via the LIVE v1 endpoint (the v0
        collection is 410-dead). This is the cost-safety verify."""
        d = _req("GET", f"{V1}/instances/", self.key)
        return [Instance(id=int(x["id"]), status=x.get("actual_status", "?"),
                         dph=float(x.get("dph_total", 0) or 0), raw=x)
                for x in d.get("instances", [])]

    def wait_running(self, instance_id: int, *, timeout_s: float = 600,
                     poll_s: float = 10) -> Instance:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            inst = self.status(instance_id)
            if inst.status == "running":
                return inst
            msg = str(inst.raw.get("status_msg") or "")
            if any(s in msg.lower() for s in _BAD_HOST_SIGNS):
                raise HostProbeFailed(
                    f"host probe failed (instance {instance_id}): {msg[:160]}")
            if inst.status in ("error", "exited"):
                raise HostProbeFailed(
                    f"instance {instance_id} -> {inst.status}: {msg[:160]}")
            time.sleep(poll_s)
        raise TimeoutError(f"instance {instance_id} not running in {timeout_s}s")

    def logs(self, instance_id: int, tail: int = 2000) -> str:
        """Onstart/container logs: request, then poll the (auth-less) result URL."""
        rj = _req("PUT", f"{V0}/instances/request_logs/{instance_id}/",
                  self.key, {"tail": str(tail)})
        url = rj.get("result_url")
        if not url:
            return json.dumps(rj)
        for _ in range(12):
            try:
                with urllib.request.urlopen(url, timeout=20) as resp:
                    if resp.status == 200:
                        return resp.read().decode(errors="replace")
            except urllib.error.HTTPError as e:
                if e.code not in (403, 404):
                    raise
            time.sleep(2)
        return ""

    def destroy(self, instance_id: int) -> None:
        _req("DELETE", f"{V0}/instances/{instance_id}/", self.key, {})

    def dead_reason(self, instance_id: int | str) -> str | None:
        """A non-None reason if this instance has VISIBLY failed -- it flipped to
        error/exited, or its status_msg matches a known bad-host signature -- else
        None (still coming up, or a transient status hiccup we shouldn't fail
        over on). Lets a readiness loop fast-fail a corpse (a container that never
        came up) in seconds instead of ssh-polling it for the full deadline (#27).
        Keeps the bad-host string matching here in the adapter so an executor's
        fast-fail stays provider-agnostic (it just asks `dead_reason`).

        Honest limit: the seconds-not-deadline speedup only holds for an
        ENUMERATED failure (a status flip, or a `_BAD_HOST_SIGNS` match). A host
        that dies for a reason not in that list returns None here and still costs
        the readiness loop its full timeout -- the list is a fast path, not total
        coverage."""
        try:
            inst = self.status(int(instance_id))
        except Exception:
            return None                       # transient status read: don't fail over
        if inst.status in ("error", "exited"):
            return f"instance {instance_id} -> {inst.status}"
        msg = str(inst.raw.get("status_msg") or "")
        if any(s in msg.lower() for s in _BAD_HOST_SIGNS):
            return f"bad host (instance {instance_id}): {msg[:160]}"
        return None

    # -- the safety primitive (the Provider F invariant) ---------------------
    @contextlib.contextmanager
    def rent(self, offer: Offer, launch: LaunchSpec, *, timeout_s: float = 600):
        """Rent `offer`, wait until usable, yield a `RentedHost`, and ALWAYS
        verify-teardown on exit -- the leak-proof contract every Provider owes.

        Destroys in a finally block on any exit -- normal, exception, or
        Ctrl-C (retried) -- then independently re-checks the v1 list. If the
        destroy failed, or the instance is still present, or teardown could not
        be confirmed, it is recorded in the ledger and a confirmed leak / failed
        destroy raises `VastError` rather than passing silently -- a leaked GPU
        bills by the second. `offer` is an Offer (not a bare id) so cost and geo
        land in the ledger. Raises `HostProbeFailed` if the host never comes up
        usable, so the executor can fail over to the next offer.
        """
        t0 = time.monotonic()
        # create runs BEFORE the try/finally: if it fails no instance exists, so
        # there is nothing to tear down. A create failure is an offer race (taken
        # between offers() and rent()) or a transient that outlived _req's retry;
        # surface it as RentUnavailable so the executor fails over provider-
        # agnostically, never as a leak.
        try:
            instance_id = self.create(offer.id, image=launch.image,
                                       onstart_cmd=launch.onstart,
                                       disk=launch.disk_gb, label=launch.label)
        except VastError as e:
            # An auth/permission failure (401/403) is a TERMINAL config error -- a
            # bad API key would otherwise be disguised as an offer race and burn
            # the whole pool into a misleading NO_OFFERS. Surface those; only an
            # ambiguous/availability failure (404 ask-gone, 5xx-after-retry,
            # transport) -- which created no instance -- becomes a failover signal.
            if e.code in (401, 403):
                raise
            raise RentUnavailable(f"could not rent offer {offer.id}: {e}") from e
        self._log("rented", provider=self.name, offer_id=offer.id,
                  instance_id=instance_id, gpu=offer.gpu_name, dph=offer.dph,
                  reliability=offer.reliability,
                  geo=offer.geolocation.strip(", "))
        outcome, reason = "ok", ""
        try:
            inst = self.wait_running(instance_id, timeout_s=timeout_s)
            # The contract is a reachable host: a "running" instance whose status
            # payload still lacks SSH coordinates is unusable. Fail it as a probe
            # failure so the executor fails over, rather than yielding a host that
            # silently breaks every downstream SSH call (empty host / port 0).
            ssh_host = str(inst.raw.get("ssh_host") or "")
            ssh_port = int(inst.raw.get("ssh_port") or 0)
            if not ssh_host or not ssh_port:
                raise HostProbeFailed(
                    f"instance {instance_id} running but SSH coordinates missing "
                    f"(host={ssh_host!r}, port={ssh_port})")
            self._log("running", provider=self.name, offer_id=offer.id,
                      instance_id=instance_id,
                      provision_s=round(time.monotonic() - t0, 1))
            yield RentedHost(
                id=str(instance_id), ssh_host=ssh_host, ssh_port=ssh_port,
                offer=offer, raw=inst.raw)
        except HostProbeFailed as e:
            outcome, reason = "host_failed", str(e)
            raise
        except TimeoutError as e:
            outcome, reason = "timeout", str(e)
            raise
        except BaseException as e:               # propagate, but record the cause
            outcome, reason = type(e).__name__, str(e)[:200]
            raise
        finally:
            billed_s = time.monotonic() - t0
            destroyed = False
            for _ in range(5):
                try:
                    self.destroy(instance_id)
                    destroyed = True
                    break
                except VastError as e:
                    # 404/410 == already gone == the DESIRED state, not a failure:
                    # `destroy` -> `_req` raises VastError on a terminal HTTP error,
                    # and a blind catch-all would retry four more times against a
                    # nonexistent instance and then raise a spurious LEAK RISK even
                    # though the box is correctly gone. Classify on `.code`, the way
                    # reap._classify does (CM review #48).
                    if e.code in (404, 410):
                        destroyed = True
                        break
                    time.sleep(2)                       # other API error: retry
                except Exception:
                    time.sleep(2)                       # transient transport: retry
            # Independently verify teardown, distinguishing gone / present /
            # unverifiable so a failed destroy or a failed verify can never
            # silently pass as success.
            try:
                present = any(i.id == instance_id for i in self.list_instances())
                verify = "present" if present else "gone"
            except Exception as e:
                verify = f"unverified: {e}"
            self._log("destroyed", provider=self.name, offer_id=offer.id,
                      instance_id=instance_id,
                      outcome=outcome, reason=reason[:200],
                      billed_s=round(billed_s, 1),
                      est_cost_usd=round(offer.dph * billed_s / 3600, 4),
                      destroyed=destroyed, verify=verify)
            # Raise LOUDLY on a confirmed leak or a failed destroy, as a structured
            # `LeakRisk` so a fleet flags it by TYPE, not by a "LEAK" substring.
            if not destroyed or verify == "present":
                raise LeakRisk(
                    f"LEAK RISK: instance {instance_id} not confirmed torn down "
                    f"(destroyed={destroyed}, verify={verify}) -- "
                    f"run `vastai destroy instance {instance_id}`")
            # destroy reported success but we couldn't double-check (list failed):
            # logged above, but warn louder -- a silent destroy-lie + blind verify
            # is the one narrow window that would otherwise read as a clean exit.
            if verify.startswith("unverified"):              # pragma: no cover
                print(f"  ~ instance {instance_id}: destroyed but teardown "
                      f"UNVERIFIED ({verify[:80]}) -- confirm it is gone.")


# Back-compat import alias: the class was `VastClient` before it implemented the
# `Provider` Protocol (F). The NAME survives so existing imports don't break, but
# `offers()` now takes a `HostSpec` and `rent()` a `LaunchSpec` (not loose
# kwargs) -- callers on the old signatures must update those calls.
VastClient = VastProvider
