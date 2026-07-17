"""Admission (E, P9) and the Provider (F) contract, physics-free.

Extracted from jax-solitons' test_campaign.py at the run-farm extraction: these
never touched the physics -- they exercise ProbeAdmission's probe-or-bail gates and
the leak-proof-teardown / failover invariants of a Provider over an in-memory
FakeProvider (no network, no spend).
"""

import contextlib

import pytest

from run_farm import (AdmissionError, HostProbeFailed, HostReport, HostSpec,
                      LaunchSpec, Offer, ProbeAdmission, Provider, RentedHost)


# ------------------------------------------------------------ E: admission --

def test_admission_probe_or_bail():
    """E (P9): admission rejects a host that cannot ship results, and passes a
    healthy one -- independent of the CI host's real hardware."""
    adm = ProbeAdmission(min_mem_gb=4.0, min_mbps=1.0)
    bad = HostReport(has_gpu=True, device_name="x", free_mem_gb=8.0,
                     outbound_mbps=0.0)            # zero outbound -- the case study
    good = HostReport(has_gpu=True, device_name="x", free_mem_gb=8.0,
                      outbound_mbps=100.0)
    adm.probe = lambda: bad
    with pytest.raises(AdmissionError):
        adm.guard()
    adm.probe = lambda: good
    assert adm.guard().outbound_mbps == 100.0


def test_admission_rejects_failed_probe():
    """E (P9): a host whose probe FAILED (probe_ok=False) is a hard reject even
    with require_gpu=False -- never 'runs anyway' on an unprobed host."""
    adm = ProbeAdmission(require_gpu=False, min_mem_gb=0.0, min_mbps=0.0)
    adm.probe = lambda: HostReport(
        has_gpu=False, device_name="probe-failed: boom", free_mem_gb=0.0,
        outbound_mbps=float("inf"), probe_ok=False)
    with pytest.raises(AdmissionError):
        adm.guard()


def test_admission_device_probe_paths(monkeypatch):
    """E: _device reads free memory from a GPU's memory_stats and treats an
    UNREADABLE capacity as UNKNOWN (+inf), not 0 -- so a healthy GPU is never
    falsely rejected. Both 'unknown' routes are covered: an empty memory_stats and
    a device with no memory_stats attribute at all."""
    import jax

    class FakeDev:
        platform = "gpu"
        device_kind = "FakeGPU"
        def __init__(self, stats): self._stats = stats
        def memory_stats(self): return self._stats

    class FakeDevNoStats:
        platform = "gpu"
        device_kind = "FakeGPU-nostats"

    monkeypatch.setattr(jax, "devices", lambda: [FakeDev(
        {"bytes_limit": 10_000_000_000, "bytes_in_use": 2_000_000_000})])
    r = ProbeAdmission(min_mem_gb=4.0).probe()
    assert r.has_gpu and r.probe_ok and abs(r.free_mem_gb - 8.0) < 0.1
    ProbeAdmission(min_mem_gb=4.0).guard()            # admits: 8 GB >= 4

    monkeypatch.setattr(jax, "devices", lambda: [FakeDev({})])
    r2 = ProbeAdmission(min_mem_gb=4.0).probe()
    assert r2.has_gpu and r2.free_mem_gb == float("inf")
    ProbeAdmission(min_mem_gb=4.0).guard()            # admits: unknown != blocked

    monkeypatch.setattr(jax, "devices", lambda: [FakeDevNoStats()])
    r3 = ProbeAdmission(min_mem_gb=4.0).probe()
    assert r3.has_gpu and r3.free_mem_gb == float("inf")
    ProbeAdmission(min_mem_gb=4.0).guard()


def test_admission_probe_exception_is_hard_reject(monkeypatch):
    """E (P9): if the device query itself throws, _device reports probe_ok=False
    and guard() hard-rejects -- even with require_gpu=False."""
    import jax
    def boom():
        raise RuntimeError("no driver")
    monkeypatch.setattr(jax, "devices", boom)
    r = ProbeAdmission(require_gpu=False).probe()
    assert r.probe_ok is False and r.device_name.startswith("probe-failed")
    with pytest.raises(AdmissionError):
        ProbeAdmission(require_gpu=False).guard()


# --------------------------------------------------------- F: Provider (P9, P10) --

def _offer(oid, *, dph=0.12, rel=0.99, cuda=12.4):
    return Offer(id=oid, dph=dph, gpu_name="RTX 3090", num_gpus=1,
                 reliability=rel, inet_down_mbps=800.0, cuda_max=cuda,
                 geolocation="Nowhere", provider="fake")


class FakeProvider:
    """In-memory `Provider` for contract tests: no network, no spend.

    Tracks a `live` set so a test can assert teardown actually fired, and
    `created` to count rental attempts. `bad_ids` come up as failed hosts
    (HostProbeFailed) -- the P9 'host lies' case the executor fails over.
    """

    name = "fake"

    def __init__(self, offers, bad_ids=()):
        self._offers = list(offers)
        self._bad = set(bad_ids)
        self.live: set[str] = set()
        self.created: list[str] = []
        self._n = 0

    def offers(self, spec: HostSpec) -> list[Offer]:
        return sorted(
            (o for o in self._offers
             if o.dph <= spec.max_dph and o.reliability >= spec.min_reliability
             and o.cuda_max >= spec.min_cuda and o.gpu_name == spec.gpu_name),
            key=lambda o: o.dph)

    @contextlib.contextmanager
    def rent(self, offer: Offer, launch: LaunchSpec, *, timeout_s: float = 600):
        iid = f"inst-{self._n}"
        self._n += 1
        self.created.append(iid)
        self.live.add(iid)                # meter "on"
        try:
            if offer.id in self._bad:
                raise HostProbeFailed(f"fake bad host {offer.id}")
            yield RentedHost(id=iid, ssh_host="fake.host", ssh_port=22, offer=offer)
        finally:
            self.live.discard(iid)        # teardown -- the F invariant


_SPEC = HostSpec(gpu_name="RTX 3090", max_dph=0.30, min_reliability=0.95,
                 min_cuda=12.2)
_LAUNCH = LaunchSpec(image="img:12.2", onstart="echo hi", disk_gb=24.0)


def test_fake_provider_satisfies_protocol():
    assert isinstance(FakeProvider([]), Provider)


def test_provider_offers_filter_and_order():
    """F discovery: offers() honours the HostSpec bar and returns cheapest-first;
    the min_cuda gate (P10) drops a host whose driver is older than the image."""
    fake = FakeProvider([
        _offer("cheap-but-old", dph=0.08, cuda=12.0),   # dropped: cuda < 12.2
        _offer("pricey", dph=0.25),
        _offer("cheap-ok", dph=0.10),
        _offer("too-dear", dph=0.40),                   # dropped: dph > max
    ])
    assert [o.id for o in fake.offers(_SPEC)] == ["cheap-ok", "pricey"]


def test_rent_yields_host_and_tears_down():
    fake = FakeProvider([_offer("a")])
    (off,) = fake.offers(_SPEC)
    with fake.rent(off, _LAUNCH) as host:
        assert isinstance(host, RentedHost)
        assert host.ssh_host == "fake.host" and host.offer.id == "a"
        assert fake.live == {host.id}
    assert fake.live == set()


def test_rent_tears_down_on_exception():
    fake = FakeProvider([_offer("a")])
    (off,) = fake.offers(_SPEC)
    with pytest.raises(ValueError):
        with fake.rent(off, _LAUNCH):
            raise ValueError("run blew up")
    assert fake.live == set()


def test_rent_bad_host_raises_and_tears_down():
    fake = FakeProvider([_offer("bad")], bad_ids={"bad"})
    (off,) = fake.offers(_SPEC)
    with pytest.raises(HostProbeFailed):
        with fake.rent(off, _LAUNCH):
            pass
    assert fake.live == set()


def test_provider_failover_pattern():
    """The failover idiom a ProviderExecutor uses over any Provider: cheapest-first,
    HostProbeFailed -> next, teardown guaranteed on every attempt."""
    fake = FakeProvider(
        [_offer("a", dph=0.10), _offer("b", dph=0.11), _offer("c", dph=0.12)],
        bad_ids={"a", "b"})
    ran = None
    for off in fake.offers(_SPEC):
        try:
            with fake.rent(off, _LAUNCH) as host:
                ran = host.offer.id
                break
        except HostProbeFailed:
            continue
    assert ran == "c"
    assert fake.live == set()
    assert fake.created == ["inst-0", "inst-1", "inst-2"]
