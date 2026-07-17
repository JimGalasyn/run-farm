"""Reap orphaned Vast instances -- the "clean up my farm" button.

The teardown-verify ``rent()`` contract only protects the exit paths it can intercept. A SIGKILL,
a crash, or a teardown REST call that itself fails on a flaky resolver all leave
GPUs billing by the second. This is the external recovery: list what's actually
live (the v1 endpoint -- the cost-safety source of truth) and destroy it, with
retry on transient errors so a network blip can't strand the cleanup the way it
stranded the rental. Destroy is idempotent: an already-gone instance counts as
success (that's the desired state), not a failure.

Scopes (--ledger is safe under concurrent farming; --label and --all need care):
  - --ledger PATH (recommended): only instances this campaign rented but never
    recorded destroyed (``rented``/``running`` minus ``destroyed``), intersected
    with what's actually still live -- safe when several sessions share one Vast
    account, since it won't touch instances this ledger never created.
  - --label NAME: instances stamped with that ``LaunchSpec.label`` at create. Safe
    only ACROSS labels -- it won't touch other labels' boxes -- but a label alone
    cannot tell an orphan from an in-use box, so ``--label`` ALONE will destroy an
    ACTIVE same-label campaign's boxes mid-run. Make it safe with an orphan
    filter: combine with --ledger (leaks only) or --older-than (age proxy). The
    raw unfiltered "destroy every box with this label, in-use included" needs the
    explicit --include-in-use, mirroring how --all gates the all-account destroy.
  - --older-than DUR: a MODIFIER, not a scope -- narrows any scope to instances
    older than DUR (e.g. ``6h``/``30m``/``2d``). Farms are short-lived, so age is
    a decent orphan proxy when no ledger is at hand. (Account-wide age-reaping
    still needs --all, e.g. ``--all --older-than 6h``.)
  - --all: EVERY live instance on the account (the "clean slate" button). With
    concurrent farming this also kills other sessions' boxes -- hence opt-in.

SAFE BY DEFAULT: a bare run only lists (dry run). Destroying needs --yes; an
unscoped destroy needs --all; an unfiltered --label destroy needs --include-in-use.

  python -m run_farm.reap                                  # list (dry run)
  python -m run_farm.reap --ledger out/vast_ledger.jsonl --yes
  python -m run_farm.reap --label eps-farm --older-than 6h --yes  # safe
  python -m run_farm.reap --label eps-farm --include-in-use --yes # in-use too
  python -m run_farm.reap --all --yes                      # nuke everything
"""
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

# instance start-time fields seen in a provider's raw record (Vast: start_date,
# epoch seconds). First present one wins; absent -> unknown age (never age-reaped).
_AGE_FIELDS = ("start_date", "created_at", "start_time")
_DUR_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

# substrings that classify a destroy error -- the FALLBACK only, used when the
# exception carries no structured status code (a provider error keys off its
# numeric `.code`, e.g. VastError.code = HTTP status; CM review #48).
_GONE = ("404", "410", "not found", "notfound", "no_such", "does not exist")
_AUTH = ("401", "403", "forbidden", "unauthor")

# Python programming errors: a destroy that raises one of these has a BUG in the
# adapter (a malformed record, a bad attr), not a network blip. Retrying just
# burns backoff and then reports a clean "failed" you'd needlessly re-run, masking
# the real fault -- so classify them as terminal and surface them (CM review #48).
_BUG_TYPES = (TypeError, KeyError, AttributeError, IndexError, NameError, ValueError)


def leaked_ids(ledger_path: str | Path) -> set[str]:
    """Instance ids a ledger rented/saw-running but never CONFIRMED destroyed.

    Pure (no network): the suspect set to intersect with what's actually live.
    A ``destroyed`` event only clears a leak when it is a *confirmed* teardown
    (``verify == "gone"``) -- the leak-proof rent() also logs a ``destroyed``
    event when teardown FAILED (``verify == "present"`` / ``destroyed: false``),
    and counting those as gone would make ledger-scoped reaping miss exactly the
    leaks it exists to catch. Erring toward "still leaked" is safe: reap()
    intersects this set with what's actually live, so a box that really is gone
    just won't be a target.

    Ids are kept as **strings** so the diff is provider-agnostic -- Vast ids are
    numeric, RunPod pod ids are not -- and `reap` compares `str(instance.id)`
    against this set. A ledger is single-provider (one campaign), so there is no
    cross-provider mixing to filter out.
    """
    seen: set[str] = set()
    confirmed_gone: set[str] = set()
    p = Path(ledger_path)
    if not p.exists():
        return set()
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        iid = ev.get("instance_id")
        if iid is None:
            continue
        iid = str(iid)
        evt = ev.get("event")
        if evt in ("rented", "running"):
            seen.add(iid)
        elif evt == "destroyed" and ev.get("verify") == "gone":
            confirmed_gone.add(iid)                     # only a verified teardown clears it
    return seen - confirmed_gone


def _label_of(inst) -> str | None:
    """The campaign/run label stamped on an instance at create (`LaunchSpec.label`),
    read from the provider's raw record; None if absent. Robust to a fake/Instance
    that carries no `raw` (returns None -> never label-matched)."""
    return (getattr(inst, "raw", None) or {}).get("label")


def _instance_age_s(inst, *, now: float | None = None) -> float | None:
    """Seconds since the instance started, from its raw record, or None if no
    start time is present. An unknown-age instance returns None and is therefore
    NEVER age-reaped -- we can't confirm it's old (the safe direction). Accepts an
    epoch number or an ISO-8601 string for the start field."""
    raw = getattr(inst, "raw", None) or {}
    val = next((raw[k] for k in _AGE_FIELDS if raw.get(k) not in (None, "")), None)
    if val is None:
        return None
    now = time.time() if now is None else now
    try:
        ts = float(val)                                  # epoch seconds
    except (TypeError, ValueError):
        try:
            dt = datetime.datetime.fromisoformat(str(val).replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:                            # naive -> assume UTC, so
            dt = dt.replace(tzinfo=datetime.timezone.utc)  # age is machine-stable
        ts = dt.timestamp()
    return max(0.0, now - ts)


def _parse_duration(s: str) -> float:
    """Parse a duration like '90s' / '30m' / '6h' / '2d' (or a bare number =
    seconds) to a POSITIVE number of seconds. Raises ValueError on a malformed or
    non-positive value: a zero/negative age filter would keep every instance
    (age >= 0 always holds) AND still count as an 'orphan filter' that satisfies
    the --label gate -- reintroducing the in-use footgun this exists to prevent."""
    s = s.strip().lower()
    secs = float(s[:-1]) * _DUR_UNITS[s[-1]] if s and s[-1] in _DUR_UNITS else float(s)
    if not secs > 0:                                     # rejects 0, negative, NaN
        raise ValueError(f"duration must be positive, got {s!r}")
    return secs


def _classify(exc: Exception) -> str:
    """transient (retry) | gone (already destroyed -> success) | auth (terminal) |
    bug (programming error -> terminal, surfaced).

    Prefers a STRUCTURED status code -- a provider error carries `.code` (VastError
    = HTTP status) -- over matching substrings in the stringified exception, where a
    bare ``404`` could appear in an instance id, a timestamp, or a request id and
    misclassify a live (still-billing) box as gone. Substrings are the fallback for
    errors with no code (CM review #48)."""
    code = getattr(exc, "code", None)
    if isinstance(code, int):
        if code in (404, 410):
            return "gone"
        if code in (401, 403):
            return "auth"
        return "transient"                              # 5xx / 408 / 429 / other
    if isinstance(exc, _BUG_TYPES):
        return "bug"                                    # a real fault, won't self-heal
    s = str(exc).lower()
    if any(t in s for t in _GONE):
        return "gone"
    if any(t in s for t in _AUTH):
        return "auth"
    return "transient"


def _destroy_with_retry(provider, iid, retries: int = 4) -> str:
    """Idempotent destroy. Returns 'destroyed' | 'gone' (already absent, success)
    | 'failed'. Retries only TRANSIENT errors; an already-gone instance is the
    desired state (no wasted backoff), and auth/permission fails fast."""
    for attempt in range(retries):
        try:
            provider.destroy(iid)
            return "destroyed"
        except Exception as e:  # noqa: BLE001 -- classify, then retry/skip/fail
            kind = _classify(e)
            if kind == "gone":
                return "gone"                       # idempotent: already destroyed
            if kind == "auth":
                print(f"  ! instance {iid}: destroy refused (auth/permission): "
                      f"{str(e)[:120]}")
                return "failed"                     # terminal -- don't burn backoff
            if kind == "bug":
                print(f"  ! instance {iid}: destroy hit a likely BUG, not retrying: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                return "failed"                     # terminal -- a retry won't fix code
            if attempt == retries - 1:
                print(f"  ! instance {iid}: destroy FAILED after {retries} tries: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                return "failed"
            time.sleep(2 ** attempt)
    return "failed"


def reap(provider, *, ledger: str | Path | None = None,
         label: str | None = None, older_than: float | None = None,
         dry_run: bool = True, retries: int = 4, live=None) -> dict:
    """List live instances, destroy the targeted ones (unless dry_run).

    provider: anything with ``list_instances() -> [Instance(id,status,dph,raw)]``
    and ``destroy(id)`` (a VastProvider, a RunPodProvider, or a fake in tests).
    Pass ``live`` to reuse an already-fetched listing (avoids a 2nd API call +
    TOCTOU). Ids are matched as **strings** (Vast numeric, RunPod not) but the
    report and the destroy call use each instance's native id.

    Filters are ANDed (each only narrows the target set, never widens it):
    ``ledger`` keeps this campaign's leaked-and-still-live instances; ``label``
    keeps instances stamped with that ``LaunchSpec.label``; ``older_than``
    (seconds) keeps only instances at least that old (an instance of unknown age
    is dropped -- never age-reaped). With no filter, the scope is every live
    instance. Returns a report dict; ``destroyed`` and ``gone`` are both successes
    (in the desired state), ``failed`` is the only error bucket.
    """
    if live is None:
        live = provider.list_instances()
    # TOCTOU: --label / --older-than target the CURRENT live set, but `live` may be
    # a snapshot main() fetched before printing every instance + running two safety
    # gates -- a box can finish and a new one be rented into the same label in that
    # gap. For a real destroy on those age/label scopes, re-fetch so we can't target
    # a box that was recycled. (--ledger is self-protecting: a recycled box gets a
    # NEW id absent from this ledger's leaked set, so its snapshot needs no refresh;
    # --all targets everything regardless, so staleness can't mis-target it.)
    if not dry_run and ledger is None and (label is not None or older_than is not None):
        live = provider.list_instances()
    targets = list(live)
    if ledger is not None:
        leaked = leaked_ids(ledger)
        targets = [i for i in targets if str(i.id) in leaked]
    if label is not None:
        targets = [i for i in targets if _label_of(i) == label]
    if older_than is not None:
        kept = []
        for i in targets:
            age = _instance_age_s(i)
            if age is not None and age >= older_than:     # unknown age -> dropped
                kept.append(i)
        targets = kept
    dph_targeted = sum(float(getattr(i, "dph", 0) or 0) for i in targets)

    report = dict(live=len(live), targeted=len(targets), destroyed=[], gone=[],
                  failed=[], dph_reclaimed=dph_targeted, dry_run=dry_run)
    if dry_run or not targets:
        return report                               # dry-run: dph_reclaimed = potential
    # actual reclaim sums ONLY confirmed-cleared boxes: a destroy that FAILED leaves
    # the box billing, so counting its dph would overstate savings exactly in the
    # partial-failure run where the number most needs to be honest (CM review #48).
    reclaimed = 0.0
    for i in targets:
        status = _destroy_with_retry(provider, i.id, retries=retries)
        report[status].append(i.id)                # native id (int vast / str runpod)
        if status in ("destroyed", "gone"):
            reclaimed += float(getattr(i, "dph", 0) or 0)
    report["dph_reclaimed"] = reclaimed
    return report


def _make_provider(name: str):
    """Instantiate a Provider by name -- the multi-cloud reaper's seam. Imports
    lazily so reaping Vast doesn't require RunPod's module (or vice versa)."""
    if name == "vast":
        from run_farm.vast import VastProvider
        return VastProvider()
    if name == "runpod":
        from run_farm.runpod import RunPodProvider
        return RunPodProvider()
    raise ValueError(f"unknown --provider {name!r} (expected: vast, runpod)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reap orphaned cloud instances.")
    ap.add_argument("--provider", default="vast", choices=("vast", "runpod"),
                    help="which cloud to reap (default: vast). Modal is serverless "
                         "-- nothing to reap.")
    ap.add_argument("--ledger", default=None,
                    help="RECOMMENDED scope: only reap instances this ledger "
                         "leaked (rented/running minus destroyed), still live")
    ap.add_argument("--label", default=None,
                    help="scope: instances with this LaunchSpec.label. A label "
                         "alone can't tell an orphan from an in-use box -- pair "
                         "with --ledger or --older-than, or pass --include-in-use")
    ap.add_argument("--older-than", default=None,
                    help="modifier: only instances older than this (6h/30m/2d or "
                         "seconds) -- an age-based orphan proxy for label scopes")
    ap.add_argument("--include-in-use", action="store_true",
                    help="allow an UNFILTERED --label destroy to kill in-use boxes "
                         "too (the dangerous explicit op; mirrors --all's gate)")
    ap.add_argument("--all", action="store_true", dest="all_scope",
                    help="scope = EVERY live instance on the account (also kills "
                         "other sessions' boxes); required for an unscoped destroy")
    ap.add_argument("--yes", action="store_true",
                    help="actually destroy (default is a dry-run listing)")
    ap.add_argument("--retries", type=int, default=4)
    args = ap.parse_args(argv)
    try:
        older_than = _parse_duration(args.older_than) if args.older_than else None
    except ValueError as e:
        print(f"invalid --older-than {args.older_than!r}: {e}")
        return 2

    provider = _make_provider(args.provider)

    live = provider.list_instances()                # fetch ONCE, reuse below
    # `is not None` (not truthiness) so the displayed scope matches the actual
    # filtering: an explicit empty `--label ''` / `--ledger ''` IS a scope (the
    # safety gate + reap() treat it as one), so it must not print as all-account.
    scope = " & ".join(
        ([f"ledger {args.ledger!r}"] if args.ledger is not None else [])
        + ([f"label {args.label!r}"] if args.label is not None else [])
        + ([f"older-than {args.older_than}"] if older_than is not None else [])
    ) or "ALL live instances"
    print(f"reap scope: {scope}")
    if not live:
        print("no live instances -- nothing to reap."); return 0
    for i in live:
        lbl = _label_of(i)
        age = _instance_age_s(i)
        tag = f"  label={lbl!r}" if lbl is not None else ""   # show even an empty label
        tag += f"  age={age / 3600:.1f}h" if age is not None else ""
        print(f"  instance {i.id}  status={i.status}  ${float(i.dph or 0):.4f}/hr{tag}")

    # safety gate: an unscoped (all-account) destroy must be explicit. A --ledger
    # scope is safe under concurrent farming, so it needs no --all.
    if (args.yes and args.ledger is None and args.label is None
            and not args.all_scope):
        print("\nREFUSING unscoped destroy: this would target ALL "
              f"{len(live)} live instances, including any concurrent session's "
              "boxes. Re-run with --ledger <path> / --label <name> (safe) or "
              "--all (clean slate).")
        return 2

    # #39 gate: a --label destroy with NO orphan filter (no --ledger, no
    # --older-than) would kill an ACTIVE same-label campaign's boxes mid-run, not
    # just orphans. Require an explicit --include-in-use, mirroring --all.
    if (args.yes and args.label is not None and args.ledger is None
            and older_than is None and not args.include_in_use):
        print("\nREFUSING unfiltered --label destroy: a label alone can't tell an "
              "orphan from an in-use box, so this would kill an ACTIVE campaign "
              f"using label {args.label!r} mid-run. Add --ledger <path> (leaks "
              "only) or --older-than <dur> (age proxy) to target orphans, or pass "
              "--include-in-use to destroy ALL boxes with this label.")
        return 2

    rep = reap(provider, ledger=args.ledger, label=args.label, older_than=older_than,
               dry_run=not args.yes, retries=args.retries, live=live)
    print(f"\nlive={rep['live']} targeted={rep['targeted']} "
          f"(~${rep['dph_reclaimed']:.3f}/hr)")
    scoped = args.ledger is not None or args.label is not None
    if rep["dry_run"]:
        if rep["targeted"]:
            scope_note = "" if scoped else "  [ALL-ACCOUNT scope]"
            print(f"DRY RUN -- pass --yes to destroy {rep['targeted']} "
                  f"instance(s).{scope_note}")
        return 0
    if not scoped:
        print("!! ALL-ACCOUNT scope: destroying every live instance !!")
    done = rep["destroyed"] + rep["gone"]
    print(f"cleared {len(done)} ({len(rep['destroyed'])} destroyed, "
          f"{len(rep['gone'])} already gone): {done}")
    if rep["failed"]:
        print(f"FAILED {len(rep['failed'])}: {rep['failed']} -- re-run to retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
