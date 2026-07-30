"""Uniform host diagnostics across providers, with the GAPS made explicit.

Providers do not have the same surface. Today `VastProvider` has `logs()` and
`RunPodProvider` does not; both have `dead_reason` and `status`, and each spells
teardown differently. That asymmetry is fine -- pretending it does not exist is not.

The failure this module exists to prevent: a monitor called `RunPodProvider.logs()`,
which does not exist, inside a `try/except` with stderr suppressed. It reported
nothing, and reporting nothing looked exactly like a healthy quiet host. A broken
check is worse than no check, because it consumes the attention a real check would
have earned.

So the contract here is: **collect what this provider can give, and NAME what it
cannot.** A `Diagnostics` with empty `logs` and an empty `gaps` list means "the host
genuinely produced no logs". A `Diagnostics` with empty `logs` and a gap saying
`logs: runpod exposes no logs()` means "I could not look." Those are different
statements and the type keeps them different.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class Gap:
    """Something diagnostics could not obtain, and why. `unsupported` separates a
    provider that cannot do this at all from a call that was tried and failed."""

    field: str
    reason: str
    unsupported: bool = False

    def __str__(self) -> str:
        kind = "unsupported" if self.unsupported else "failed"
        return f"{self.field} ({kind}): {self.reason}"


@dataclasses.dataclass(frozen=True)
class Diagnostics:
    """What could be learned about one host, plus what could not.

    Read `gaps` before drawing any conclusion from a None/empty field: absence of a
    value here does not mean absence of the condition.
    """

    host_id: str
    provider: str
    alive: bool | None = None          # None = could not determine
    status: Any = None
    dead_reason: str | None = None
    logs: str | None = None
    gaps: tuple[Gap, ...] = ()

    @property
    def complete(self) -> bool:
        """True when every requested field was obtained. When False, any negative
        conclusion ("no logs", "not dead") is unsupported by the evidence."""
        return not self.gaps

    def summary(self) -> str:
        bits = [f"host {self.host_id} @{self.provider}"]
        if self.alive is not None:
            bits.append("ALIVE" if self.alive else "not alive")
        if self.dead_reason:
            bits.append(f"dead_reason={self.dead_reason!r}")
        if self.logs is not None:
            n = len(self.logs.splitlines())
            bits.append(f"{n} log line(s)")
        if self.gaps:
            bits.append("COULD NOT DETERMINE: "
                        + "; ".join(str(g) for g in self.gaps))
        return " | ".join(bits)

    def __str__(self) -> str:
        return self.summary()


def _try(field: str, fn, gaps: list[Gap], *args, **kw):
    """Call `fn`, recording a Gap instead of swallowing the failure.

    The whole point: an exception here becomes visible evidence, not silence. Note
    `AttributeError` is NOT special-cased into "unsupported" -- an unsupported
    capability is detected by absence before the call (see `collect`), so an
    AttributeError raised *inside* a real method is a genuine bug and is reported as
    a failure, not excused as a missing feature.
    """
    try:
        return fn(*args, **kw)
    except Exception as e:                                    # noqa: BLE001
        gaps.append(Gap(field, f"{type(e).__name__}: {e}"))
        return None


def capabilities(provider) -> dict[str, bool]:
    """Which diagnostic capabilities this provider actually exposes.

    Call it at launch (see `gauntlet.ProviderCapable`) so a thin provider is a known
    condition rather than a surprise three hours into a run."""
    return {m: callable(getattr(provider, m, None))
            for m in ("status", "dead_reason", "logs", "list_instances", "destroy")}


def collect(provider, host_id: str, *, want_logs: bool = True,
            log_tail: int = 2000) -> Diagnostics:
    """Gather everything `provider` can say about `host_id`.

    Never raises for a provider limitation or a failed call: both are recorded in
    `gaps`. Raising here would be its own trap -- diagnostics run on the error path,
    and a diagnostic that throws destroys the report you were trying to write.
    """
    pname = getattr(provider, "name", type(provider).__name__)
    caps = capabilities(provider)
    gaps: list[Gap] = []

    status = None
    if caps["status"]:
        status = _try("status", provider.status, gaps, host_id)
    else:
        gaps.append(Gap("status", f"{pname} exposes no status()", unsupported=True))

    dead = None
    if caps["dead_reason"]:
        dead = _try("dead_reason", provider.dead_reason, gaps, host_id)
    else:
        gaps.append(Gap("dead_reason", f"{pname} exposes no dead_reason()",
                        unsupported=True))

    logs = None
    if want_logs:
        if caps["logs"]:
            logs = _try("logs", provider.logs, gaps, host_id, log_tail)
        else:
            gaps.append(Gap("logs", f"{pname} exposes no logs() -- host-side stdout "
                                    "is not retrievable through this adapter",
                            unsupported=True))

    # `alive` is deliberately three-valued. A provider that cannot list instances
    # yields None ("could not determine"), never False -- reporting "not alive" on
    # the strength of a call that never happened is how a healthy host gets reaped.
    alive: bool | None = None
    if caps["list_instances"]:
        listed = _try("list_instances", provider.list_instances, gaps)
        if listed is not None:
            ids = {str(getattr(i, "id", getattr(i, "pod_id", i))) for i in listed}
            alive = str(host_id) in ids
    else:
        gaps.append(Gap("alive", f"{pname} exposes no list_instances()",
                        unsupported=True))

    if dead is not None and not isinstance(dead, str):        # normalise
        dead = str(dead)

    return Diagnostics(host_id=str(host_id), provider=pname, alive=alive,
                       status=status, dead_reason=dead, logs=logs,
                       gaps=tuple(gaps))


def explain_failure(provider, host_id: str, leg_detail: str = "") -> str:
    """A human-readable account of why a leg on `host_id` may have failed.

    Written for the log line that a person reads at 2am. It states the evidence it
    HAS, and then states what it could not check -- because "exit=1" with no further
    information is exactly the log that cost a rental's worth of guessing.
    """
    d = collect(provider, host_id)
    lines = [f"leg failure on {d.host_id} @{d.provider}"]
    if leg_detail:
        lines.append(f"  leg said: {leg_detail[:300]}")
    if d.dead_reason:
        lines.append(f"  the provider CONFIRMS the host died: {d.dead_reason}")
        lines.append("  -> a host failure, not a work failure; the leg is retryable")
    elif d.alive is True:
        lines.append("  the host is still alive, so this is a genuine WORK failure "
                     "-- re-running it on fresh hardware will fail the same way")
    elif d.alive is None:
        lines.append("  could not determine whether the host is alive; do not "
                     "conclude either way from this report")
    if d.logs:
        tail = [ln for ln in d.logs.splitlines() if ln.strip()][-8:]
        lines.append("  last log lines:")
        lines += [f"    {ln[:200]}" for ln in tail]
    for g in d.gaps:
        lines.append(f"  NOT CHECKED -- {g}")
    return "\n".join(lines)
