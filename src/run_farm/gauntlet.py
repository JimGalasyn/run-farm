"""The launch gauntlet: everything that can fail BEFORE any money is spent.

A campaign's expensive failures are almost never physics. They are a missing local
file, a key that was never registered, a payload one file short, a stale marker that
made a skipped leg read as a passing one. Each of those is detectable for $0, and each
of them has instead been detected by renting GPUs.

The governing rule, learned the hard way: **every check must be able to fail. If it
cannot, it is not a check.** The canonical violation cost 7 rentals and 72 minutes --
a Vast API call (`list_instances()`) stood in for an SSH test. The API key was fine.
`~/.ssh/vastai` did not exist. Every host rejected the connection, and each rejection
looked like a spot-pool failure (TimeoutError / HostProbeFailed) rather than the one
local cause it was.

That is why `CheckResult` carries `proves`: a passing check must state what its green
result actually establishes, and what it does NOT. A check that cannot articulate the
difference is the API-call-for-SSH-test mistake waiting to happen again.

What the gauntlet can and cannot do, stated plainly:

  CAN   prove a key file exists and is offerable, that its fingerprint is registered
        with the provider, that the payload runs when flattened, that the marketplace
        has capacity, that the output dir is writable, and that no leg is about to
        pre-skip on a marker you did not intend.
  CANNOT prove SSH works. A real handshake needs a host, and a host costs money. The
        strongest pre-rental statement available is "the key ssh will offer is one the
        provider has on file" -- which is precisely the statement whose absence caused
        the 72-minute failure. Use `SshHandshake` once you actually hold a host (a
        smoke leg) for the rest.

This module is deliberately NOT called `preflight`. That word is already taken, by a
different layer that runs on a different object, and conflating them would be the kind
of confusion that hides a skipped check:

  `gauntlet` (here)            the LOCAL LAUNCH ENVIRONMENT. Is the key there, does
                               the payload run flat, has the provider capacity, will
                               a leg skip when you did not mean it to. Knows nothing
                               about physics.
  `FarmCampaign(preflight=)`   the DOMAIN ENVELOPE, injected per engine, run against
                               a config dict: can this configuration hold at all?
                               ("don't pay for a config that can't hold.")
  `farm.launch_gate`           COMPLETENESS: the launched config-hash set equals the
                               planned one, so no leg was silently dropped.

All three run around launch and none substitutes for another: a physically sound
config still fails at `exit=1` if the payload is a file short, a perfectly closed
payload still wastes a rental computing a config outside its envelope, and both can
be fine while a third of the legs never launched at all.
"""

from __future__ import annotations

import dataclasses
import os
import stat
import subprocess
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from run_farm.payload import PayloadSpec, validate_flat


@dataclasses.dataclass(frozen=True)
class CheckResult:
    """The outcome of one gauntlet check.

    name    short stable id, e.g. "ssh-key-present"
    ok      did it pass
    detail  why -- for a failure, what to DO about it
    proves  what a PASS establishes, and explicitly what it does not. Required for
            every check: this field exists because an API-key call once stood in for
            an SSH test, and a `proves` string forces that gap into the open.
    fatal   a False+fatal check blocks launch; False+non-fatal is a loud warning
            (e.g. "this leg will be skipped") that a human may legitimately accept.
    """

    name: str
    ok: bool
    detail: str
    proves: str = ""
    fatal: bool = True

    @property
    def blocking(self) -> bool:
        return not self.ok and self.fatal

    def __str__(self) -> str:
        mark = "ok  " if self.ok else ("FAIL" if self.fatal else "warn")
        return f"[{mark}] {self.name}: {self.detail}"


class GauntletError(RuntimeError):
    """Raised by `require_gauntlet` when a fatal check failed. Carries every result,
    not just the first failure, so ONE local run surfaces every blocker."""

    def __init__(self, results: Sequence[CheckResult]):
        self.results = list(results)
        self.blocking = [r for r in self.results if r.blocking]
        super().__init__(
            f"{len(self.blocking)} gauntlet check(s) blocked launch:\n  "
            + "\n  ".join(str(r) for r in self.blocking))


def run_gauntlet(checks: Iterable, *, log=print, skip: Sequence[str] = ()) -> list[CheckResult]:
    """Run every check, in order, and return all results.

    Does NOT stop at the first failure and does NOT raise: the point is to spend one
    local run discovering everything wrong, rather than one rental per defect. A check
    that raises is itself reported as a failure -- a broken check must never read as a
    quiet pass, which is how a monitor calling a nonexistent provider method with
    stderr suppressed looked like a healthy silence.
    """
    results: list[CheckResult] = []
    for check in checks:
        name = getattr(check, "name", getattr(check, "__name__", type(check).__name__))
        if name in skip:
            # Recorded, never silent. A skipped check that vanishes from the report
            # is indistinguishable from one that passed -- which is the whole failure
            # this module exists to prevent. It is also non-fatal by construction:
            # the human took responsibility, so say so and carry on.
            r = CheckResult(name, True, "SKIPPED by caller — not verified here",
                            "nothing. A skip proves only that someone chose to "
                            "accept this risk out-of-band.", fatal=False)
            results.append(r)
            if log:
                log(str(r))
            continue
        try:
            r = check()
        except Exception as e:                                # noqa: BLE001
            r = CheckResult(name, False,
                            f"the CHECK ITSELF raised {type(e).__name__}: {e} "
                            "-- treat as a failure, not as a pass")
        results.append(r)
        if log:
            log(str(r))
    return results


def require_gauntlet(checks: Iterable, *, log=print,
                     skip: Sequence[str] = ()) -> list[CheckResult]:
    """`run_gauntlet`, but raise `GauntletError` if anything fatal failed.

    `skip` names checks to bypass, by `CheckResult.name`. Two checks tell the
    caller to use it (`ssh-key-registered` when the provider exposes no way to
    enumerate keys); before this existed those messages named a parameter that
    was not implemented, so the only way to follow the advice was to edit the
    check list -- and a caller who edits the list usually deletes the check
    rather than recording that it was bypassed.
    """
    results = run_gauntlet(checks, log=log, skip=skip)
    if any(r.blocking for r in results):
        raise GauntletError(results)
    return results


# ------------------------------------------------------------------ ssh ----
def _fingerprint(path: Path) -> str | None:
    """SHA256 fingerprint of a key/pubkey via ssh-keygen, or None."""
    try:
        r = subprocess.run(["ssh-keygen", "-lf", str(path)],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):             # pragma: no cover
        return None
    if r.returncode != 0:
        return None
    parts = r.stdout.split()
    return next((p for p in parts if p.startswith("SHA256:")), None)


class SshKeyPresent:
    """The key ssh will offer exists, is private-mode, and has a public half.

    THE check whose absence cost 7 rentals: `FleetExecutor` defaults to
    `key_path=~/.ssh/vastai`, that file did not exist, and nothing looked at it until
    every host had already been rented and had already refused the connection.
    """

    name = "ssh-key-present"

    def __init__(self, key_path: str = "~/.ssh/vastai"):
        self.key_path = key_path

    def __call__(self) -> CheckResult:
        p = Path(self.key_path).expanduser()
        proves = ("ssh has a private key to offer, and its public half is readable. "
                  "Does NOT prove the provider accepts it, and does NOT prove any "
                  "host is reachable -- a real handshake needs a rented host.")
        if not p.exists():
            return CheckResult(
                self.name, False,
                f"{p} does not exist -- every rental will refuse the connection and "
                f"each refusal will look like a bad host. Create it "
                f"(ssh-keygen -t ed25519 -f {p}) and register the public half with "
                f"the provider.", proves)
        mode = stat.S_IMODE(p.stat().st_mode)
        if mode & 0o077:
            return CheckResult(
                self.name, False,
                f"{p} has mode {mode:04o}; ssh refuses group/world-readable private "
                f"keys. chmod 600 {p}", proves)
        pub = p.with_name(p.name + ".pub")
        if not pub.exists():
            return CheckResult(
                self.name, False,
                f"{pub} is missing, so the fingerprint cannot be compared against "
                f"the provider's registered keys (ssh-keygen -y -f {p} > {pub})",
                proves)
        fp = _fingerprint(pub)
        return CheckResult(self.name, True,
                           f"{p} present, mode {mode:04o}, fingerprint {fp}", proves)


class SshKeyRegistered:
    """The local key's fingerprint is one the PROVIDER has on file.

    Requires the provider to expose `registered_ssh_keys() -> list[str]` of public-key
    strings. If it does not, this check FAILS as a capability gap rather than passing:
    "I could not look" must never render as "it is fine". That inversion is the whole
    bug class -- a check that cannot fail is decoration.
    """

    name = "ssh-key-registered"

    def __init__(self, provider, key_path: str = "~/.ssh/vastai"):
        self.provider, self.key_path = provider, key_path

    def __call__(self) -> CheckResult:
        proves = ("the key ssh will offer is one the provider has on file, so a "
                  "connection refusal is NOT a missing-registration problem. Does "
                  "NOT prove the host-side sshd is up or the network path works.")
        pub = Path(self.key_path).expanduser().with_name(
            Path(self.key_path).name + ".pub")
        mine = _fingerprint(pub)
        if mine is None:
            return CheckResult(self.name, False,
                               f"cannot fingerprint {pub} (missing or unreadable); "
                               "run the ssh-key-present check first", proves)

        lister = getattr(self.provider, "registered_ssh_keys", None)
        if lister is None:
            return CheckResult(
                self.name, False,
                f"provider {getattr(self.provider, 'name', '?')} exposes no "
                "registered_ssh_keys(), so registration CANNOT be verified here. "
                "This is a capability gap reported as a failure on purpose: an "
                "unverifiable check must not read as a pass. Confirm the key is "
                "registered in the provider console, then pass "
                "skip=('ssh-key-registered',).", proves)

        registered = list(lister())
        fps = []
        for entry in registered:
            fd, name = tempfile.mkstemp(prefix="rf_pub_", suffix=".pub")
            tmp = Path(name)
            try:
                os.close(fd)
                tmp.write_text(entry if entry.endswith("\n") else entry + "\n")
                fp = _fingerprint(tmp)
            finally:
                tmp.unlink(missing_ok=True)
            if fp:
                fps.append(fp)
        if mine in fps:
            return CheckResult(self.name, True,
                               f"{mine} is registered ({len(fps)} key(s) on file)",
                               proves)
        return CheckResult(
            self.name, False,
            f"{mine} is NOT among the {len(fps)} key(s) registered with "
            f"{getattr(self.provider, 'name', '?')}; every host will refuse the "
            f"connection. Register {pub}.", proves)


class SshHandshake:
    """A REAL handshake against a host you already hold. The only check that
    actually proves SSH works -- and it needs a host, so it cannot run pre-rental.
    Use it as the first step of a smoke leg."""

    name = "ssh-handshake"

    def __init__(self, host: str, port: int, key_path: str = "~/.ssh/vastai",
                 timeout: float = 30):
        self.host, self.port, self.key_path, self.timeout = (
            host, port, key_path, timeout)

    def __call__(self) -> CheckResult:
        from run_farm.provider_exec import _ssh
        proves = ("ssh authenticated to THIS host and ran a command. Proves key, "
                  "network path and sshd together -- the statement no pre-rental "
                  "check can make.")
        rc, out = _ssh(self.key_path, self.host, self.port, "true", self.timeout)
        if rc == 0:
            return CheckResult(self.name, True,
                               f"authenticated to {self.host}:{self.port}", proves)
        return CheckResult(self.name, False,
                           f"{self.host}:{self.port} rc={rc}: {out[-200:]}", proves)


# -------------------------------------------------------------- payload ----
class PayloadClosed:
    """The shipped files actually run when flattened into the worker cwd.

    Delegates to `run_farm.payload.validate_flat`; see that module for the three
    rental-costing failures it reconstructs."""

    name = "payload-closed"

    def __init__(self, spec: PayloadSpec, **kw):
        self.spec, self.kw = spec, kw

    def __call__(self) -> CheckResult:
        proves = ("every shipped file exists, imports resolve INSIDE the flat dir, "
                  "and the entrypoint's real startup path runs there. Does NOT prove "
                  "the box's GPU/driver/wheels match -- that is environment parity, "
                  "not payload closure.")
        problems = validate_flat(self.spec, **self.kw)
        if not problems:
            n = len(self.spec.files)
            return CheckResult(self.name, True,
                               f"{n} file(s) run flat; "
                               f"{len(self.spec.imports)} import(s) resolve locally",
                               proves)
        return CheckResult(self.name, False,
                           "; ".join(str(p) for p in problems)[:600], proves)


# ------------------------------------------------------------ marketplace ----
class OffersAvailable:
    """The marketplace has capacity at this spec, checked with a FREE call.

    Cheap insurance against launching a campaign into an empty pool and reading the
    resulting `NO_OFFERS` as a code fault."""

    name = "offers-available"

    def __init__(self, provider, host_spec, *, minimum: int = 1):
        self.provider, self.host_spec, self.minimum = provider, host_spec, minimum

    def __call__(self) -> CheckResult:
        proves = ("the provider listed rentable offers meeting the spec at this "
                  "moment. Does NOT reserve them: offers race, and one can be taken "
                  "between listing and renting (that is what RentUnavailable is).")
        offers = list(self.provider.offers(self.host_spec))
        if len(offers) >= self.minimum:
            cheap = min(o.dph for o in offers)
            return CheckResult(self.name, True,
                               f"{len(offers)} offer(s), cheapest ${cheap:.3f}/hr",
                               proves)
        return CheckResult(
            self.name, False,
            f"only {len(offers)} offer(s) meet the spec (need {self.minimum}); "
            "relax gpu_name/max_dph/min_reliability or wait for capacity", proves)


# ----------------------------------------------------------------- local ----
class OutDirWritable:
    """The local output dir can actually be written. A campaign that fetches for
    three hours into an unwritable path has thrown away three hours."""

    name = "out-dir-writable"

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def __call__(self) -> CheckResult:
        proves = ("this process can create and write files under the output dir "
                  "right now. Does NOT prove there is room for the whole campaign.")
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            probe = self.path / f".rf_write_probe_{os.getpid()}"
            probe.write_text("ok")
            probe.unlink()
        except OSError as e:
            return CheckResult(self.name, False, f"{self.path}: {e}", proves)
        return CheckResult(self.name, True, f"{self.path} writable", proves)


class RemoteEnvPinned:
    """The executor will actually ship the environment variables a campaign needs.

    For variables the worker must see AT PROCESS START. `ProviderExecutor.remote_env`
    is the only way to set those (a non-interactive `ssh host cmd` sources no profile,
    so onstart's exports do not reach the worker), and forgetting it is silent: every
    leg runs, produces plausible numbers, and only the reproducibility claim is void.

    The motivating case is `XLA_FLAGS=--xla_gpu_autotune_level=0`. XLA GPU autotuning
    selects kernels per process, so without it a resumed leg is not bit-identical to
    an uninterrupted one -- measured at campaign scale, 1 divergence in 3 attempts,
    73% of entries. A campaign that promises resume exactness while omitting this is
    making a claim it cannot keep, and nothing downstream would notice.

    Fatal by default: this guards a correctness property, not a convenience.
    """

    name = "remote-env-pinned"

    def __init__(self, executor, required: Mapping[str, str], *, fatal: bool = True):
        self.executor, self.required, self.fatal = executor, dict(required), fatal

    def __call__(self) -> CheckResult:
        proves = ("the executor is configured to ship these variables to the worker, "
                  "with these values. Does NOT prove the box honours them, that the "
                  "values are correct for this engine, or that they suffice for "
                  "reproducibility -- only that what you meant to send is set to send.")
        actual = dict(getattr(self.executor, "remote_env", {}) or {})
        missing, wrong = [], []
        for k, v in self.required.items():
            if k not in actual:
                missing.append(k)
            elif actual[k] != v:
                wrong.append(f"{k}={actual[k]!r} (wanted {v!r})")
        if not missing and not wrong:
            keys = ", ".join(sorted(self.required))
            return CheckResult(self.name, True, f"remote_env pins {keys}", proves)
        bits = []
        if missing:
            bits.append("missing: " + ", ".join(sorted(missing)))
        if wrong:
            bits.append("wrong: " + "; ".join(sorted(wrong)))
        return CheckResult(
            self.name, False,
            " | ".join(bits) + " -- pass remote_env={...} to the executor",
            proves, fatal=self.fatal)


class ResumeMarkersIntended:
    """Report every leg that will be PRE-SKIPPED, and why.

    Non-fatal by design, but loud, because of a specific failure: a leg reported
    `SKIP (output already present)` and that read as a pass -- the marker satisfying
    `done_when` had been left by a DIFFERENT PROVIDER's earlier run. A skip is a
    claim that work is already done; it deserves the same scrutiny as a result.

    Reports each marker's age, so a marker from a previous campaign is visible rather
    than inferred.
    """

    name = "resume-markers-intended"

    def __init__(self, legs, out_dir: str | Path):
        self.legs, self.out_dir = list(legs), Path(out_dir)

    def __call__(self) -> CheckResult:
        proves = ("names every leg that will be skipped and how old its marker is. "
                  "Does NOT prove a marker's contents are correct or that it came "
                  "from the campaign you think it did -- only a human can say that.")
        skipping, unmarked = [], []
        for leg in self.legs:
            marker = leg.marker()
            if not marker:
                unmarked.append(leg.label)
                continue
            path = self.out_dir / leg.label / marker
            if path.exists():
                age_h = (time.time() - path.stat().st_mtime) / 3600.0
                skipping.append(f"{leg.label} (marker {marker}, {age_h:.1f}h old)")
        bits = []
        if skipping:
            bits.append(f"{len(skipping)} leg(s) will be SKIPPED: "
                        + "; ".join(skipping)
                        + " -- confirm these came from THIS campaign and provider")
        if unmarked:
            bits.append(f"{len(unmarked)} leg(s) have no done_when marker and can "
                        f"never resume: {', '.join(unmarked)}")
        if not bits:
            return CheckResult(self.name, True,
                               f"no leg pre-skips; all {len(self.legs)} will run",
                               proves)
        return CheckResult(self.name, False, " | ".join(bits), proves, fatal=False)


# ------------------------------------------------------------ capability ----
class ProviderCapable:
    """The provider actually HAS the methods this campaign will call on it.

    From a real failure: a monitor called `RunPodProvider.logs()`, which does not
    exist (only the Vast adapter has `logs`), with stderr suppressed -- so a broken
    monitor looked like a quiet one for the length of a run. Checking the surface up
    front turns that into a launch-time message.
    """

    name = "provider-capable"

    def __init__(self, provider, required: Sequence[str] = (),
                 optional: Sequence[str] = ()):
        self.provider = provider
        self.required, self.optional = tuple(required), tuple(optional)

    def __call__(self) -> CheckResult:
        pname = getattr(self.provider, "name", type(self.provider).__name__)
        proves = (f"{pname} exposes the methods named as required, so a call to one "
                  "will not fail with AttributeError mid-run. Does NOT prove they "
                  "work or that the remote API still honours them.")
        missing = [m for m in self.required
                   if not callable(getattr(self.provider, m, None))]
        absent_opt = [m for m in self.optional
                      if not callable(getattr(self.provider, m, None))]
        if missing:
            return CheckResult(
                self.name, False,
                f"{pname} is MISSING required method(s): {', '.join(missing)} -- a "
                "call would raise AttributeError at runtime, which reads as a broken "
                "host rather than a broken caller", proves)
        detail = f"{pname} has all {len(self.required)} required method(s)"
        if absent_opt:
            detail += (f"; degraded (absent optional): {', '.join(absent_opt)} -- "
                       "diagnostics will be thinner on this provider")
        return CheckResult(self.name, True, detail, proves)


# -------------------------------------------------------------- assembly ----
class CapClearsWorstCase:
    """The budget cap is above what this campaign could COST at its own timeout.

    A cap below the worst case does not prevent overspend -- it guarantees the
    campaign dies partway through, after paying for whatever ran. The money is
    gone and there is no result, which is the one outcome worse than either
    finishing or refusing to start.

    Worst case is `n_legs * run_timeout_h * max_dph` plus acquisition tax, i.e.
    every leg running to its timeout on the priciest host the spec allows -- not
    the expected cost. That is the number the cap has to clear, because the cap
    is enforced against actual spend as it accrues, and spend accrues at the
    worst-case rate whenever things go badly.

    Non-fatal: a deliberately tight cap on a campaign you intend to babysit is a
    legitimate choice. It just must not be an accident.
    """

    name = "cap-clears-worst-case"

    def __init__(self, cap_usd, worst_case_usd, *, already_spent_usd=0.0):
        self.cap = float(cap_usd)
        self.worst = float(worst_case_usd)
        self.spent = float(already_spent_usd)

    def __call__(self) -> CheckResult:
        proves = ("the cap is above this campaign's own worst-case estimate, so a "
                  "bad run stops by finishing rather than by hitting the cap. Does "
                  "NOT bound real spend: the estimate is only as good as the "
                  "timeout and price ceiling it was computed from.")
        need = self.spent + self.worst
        if self.cap >= need:
            return CheckResult(self.name, True,
                               f"cap ${self.cap:.2f} >= ${need:.2f} "
                               f"(worst case ${self.worst:.2f}"
                               + (f" + ${self.spent:.2f} already on this ledger)"
                                  if self.spent else ")"),
                               proves)
        return CheckResult(
            self.name, False,
            f"cap ${self.cap:.2f} is BELOW the ${need:.2f} this campaign could "
            f"cost (worst case ${self.worst:.2f}"
            + (f" + ${self.spent:.2f} already on this ledger). " if self.spent
               else "). ")
            + f"It would abort mid-campaign having already paid. Raise the cap "
              f"above ${need:.2f}, or lower the run-timeout / max-dph the estimate "
              f"is computed from.",
            proves, fatal=False)


def standard_gauntlet(*, provider, host_spec, out_dir, key_path="~/.ssh/vastai",
                      payload: PayloadSpec | None = None, legs=(),
                      required_methods=("offers", "rent", "destroy"),
                      optional_methods=("dead_reason", "logs", "list_instances"),
                      minimum_offers: int = 1, skip: Sequence[str] = ()) -> list:
    """The checks worth running before every campaign, in cheapest-first order.

    Local and free before anything that touches the network, so a missing file fails
    in milliseconds rather than after an API round trip. `skip` drops checks by name
    -- use it deliberately (e.g. a provider with no `registered_ssh_keys`), not to
    quiet an inconvenient failure.
    """
    checks = [
        SshKeyPresent(key_path),
        OutDirWritable(out_dir),
        ProviderCapable(provider, required_methods, optional_methods),
    ]
    if payload is not None:
        checks.append(PayloadClosed(payload))
    if legs:
        checks.append(ResumeMarkersIntended(legs, out_dir))
    checks += [
        SshKeyRegistered(provider, key_path),
        OffersAvailable(provider, host_spec, minimum=minimum_offers),
    ]
    return [c for c in checks if getattr(c, "name", None) not in set(skip)]
