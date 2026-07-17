"""RunPodProvider lifecycle tests with a mocked HTTP layer — no network, no spend.

All HTTP goes through `runpod._req`, so a scriptable fake covers the GraphQL
gpuTypes catalog (offers) and the REST pod lifecycle (create/status/list/delete),
including the leak-proof `rent()` teardown that must surface a leak loudly.
"""

import pytest

import run_farm.runpod as runpod
from run_farm import (
    HostProbeFailed,
    HostSpec,
    LaunchSpec,
    Offer,
    Provider,
    RentedHost,
    VastLedger,
)
from run_farm.runpod import (
    GQL,
    REST,
    LeakRisk,
    RunPodError,
    RunPodProvider,
)

LAUNCH = LaunchSpec(image="img:12.2", onstart="echo hi", disk_gb=24)

_CATALOG = [
    dict(id="NVIDIA GeForce RTX 3090", displayName="RTX 3090", memoryInGb=24,
         secureCloud=True, communityCloud=True, securePrice=0.46, communityPrice=0.22),
    dict(id="NVIDIA GeForce RTX 3090 Ti", displayName="RTX 3090 Ti", memoryInGb=24,
         secureCloud=False, communityCloud=True, securePrice=0.46, communityPrice=0.27),
    dict(id="NVIDIA GeForce RTX 4090", displayName="RTX 4090", memoryInGb=24,
         secureCloud=True, communityCloud=True, securePrice=0.69, communityPrice=0.34),
]


class FakeRunPod:
    """Scriptable replacement for runpod._req, routing by (method, url)."""

    def __init__(self, *, status="RUNNING", ip="1.2.3.4", port=40022,
                 delete_fail=False, delete_noop=False, delete_404=False,
                 gql_errors=None):
        self.status, self.ip, self.port = status, ip, port
        self.delete_fail, self.delete_noop = delete_fail, delete_noop
        self.delete_404 = delete_404       # pod gone, but DELETE returns HTTP 404
        self.gql_errors = gql_errors
        self.pods: dict[str, dict] = {}    # id -> pod dict (the "live" set)
        self.last_create = None            # the create payload (for label tests)

    def _pod(self, pid, name="jax-solitons"):
        return {"id": pid, "desiredStatus": self.status, "publicIp": self.ip,
                "portMappings": {"22": self.port}, "costPerHr": 0.22, "name": name}

    def __call__(self, method, url, key, payload=None, timeout=30):
        if url == GQL:                                   # offers catalog
            if self.gql_errors:
                return {"errors": self.gql_errors}
            return {"data": {"gpuTypes": _CATALOG}}
        if method == "POST" and url == f"{REST}/pods":   # create
            self.last_create = payload
            pid = "pod1"
            self.pods[pid] = self._pod(pid, name=(payload or {}).get("name", "jax-solitons"))
            return {"id": pid}
        if method == "GET" and url == f"{REST}/pods":    # list (verify)
            return list(self.pods.values())
        if method == "GET" and "/pods/" in url:          # status
            pid = url.rstrip("/").split("/pods/")[1]
            return self.pods.get(pid, self._pod(pid))
        if method == "DELETE" and "/pods/" in url:
            if self.delete_404:                          # already gone: API 404s
                self.pods.pop(url.rstrip("/").split("/pods/")[1], None)
                err = RunPodError("DELETE /pods/.. -> HTTP 404: not found")
                err.code = 404
                raise err
            if self.delete_fail:
                raise RunPodError("terminate boom")
            if not self.delete_noop:
                pid = url.rstrip("/").split("/pods/")[1]
                self.pods.pop(pid, None)
            return {}
        return {}


@pytest.fixture
def mk(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rpa_testkey")
    monkeypatch.setattr(runpod.time, "sleep", lambda s: None)

    def make(fake, **kw):
        monkeypatch.setattr(runpod, "_req", fake)
        return RunPodProvider(**kw)
    return make


def test_runpod_provider_satisfies_protocol():
    assert isinstance(RunPodProvider(api_key="x"), Provider)


def test_offers_filters_by_tier_price_and_sorts(mk):
    p = mk(FakeRunPod(), cloud_type="COMMUNITY")
    offs = p.offers(HostSpec(gpu_name="RTX_3090", max_dph=0.25, num_gpus=1))
    # 3090 ($0.22) only — 3090 Ti is $0.27 > max; both are community-available.
    assert [(o.id, o.dph) for o in offs] == [("NVIDIA GeForce RTX 3090", 0.22)]
    o = offs[0]
    assert o.provider == "runpod" and o.geolocation == "COMMUNITY"
    assert o.num_gpus == 1                              # carried for rent's gpuCount


def test_offers_secure_tier_excludes_community_only(mk):
    """3090 Ti has communityCloud only — it must not appear on the SECURE tier."""
    p = mk(FakeRunPod(), cloud_type="SECURE")
    ids = [o.id for o in p.offers(HostSpec(gpu_name="RTX_3090", max_dph=1.0))]
    assert "NVIDIA GeForce RTX 3090 Ti" not in ids
    assert "NVIDIA GeForce RTX 3090" in ids            # secureCloud True


def test_offers_raises_on_graphql_error(mk):
    p = mk(FakeRunPod(gql_errors=[{"message": "boom"}]))
    with pytest.raises(RunPodError, match="gpuTypes"):
        p.offers(HostSpec(gpu_name="RTX_4090"))


def test_min_cuda_gate_from_spec(mk):
    """offers(spec) remembers spec.min_cuda; rent sends allowedCudaVersions >= it."""
    p = mk(FakeRunPod())
    p.offers(HostSpec(gpu_name="RTX_4090", max_dph=1.0, min_cuda=12.4))
    assert p._allowed_cuda() == ["12.4", "12.5", "12.6", "12.7", "12.8", "13.0"]


def test_offers_refreshes_gates_across_specs(mk):
    """A reused provider tracks the most recent spec's create-time gates; a
    constructor override pins them against refresh."""
    p = mk(FakeRunPod())
    p.offers(HostSpec(gpu_name="RTX_4090", max_dph=1.0, min_cuda=12.4))
    assert p._min_cuda == 12.4
    p.offers(HostSpec(gpu_name="RTX_4090", max_dph=1.0, min_cuda=12.8))
    assert p._min_cuda == 12.8                          # refreshed, not stuck at 12.4

    pinned = mk(FakeRunPod(), min_cuda=11.8)
    pinned.offers(HostSpec(gpu_name="RTX_4090", max_dph=1.0, min_cuda=12.8))
    assert pinned._min_cuda == 11.8                     # constructor wins, unchanged


def _offer():
    return Offer(id="NVIDIA GeForce RTX 4090", dph=0.34, gpu_name="RTX 4090",
                 num_gpus=1, reliability=float("nan"), inet_down_mbps=float("nan"),
                 cuda_max=float("nan"), geolocation="COMMUNITY", provider="runpod")


def test_rent_happy_path_terminates_and_verifies_gone(mk, tmp_path):
    led = VastLedger(tmp_path / "l.jsonl")
    p = mk(FakeRunPod(status="RUNNING"), ledger=led)
    with p.rent(_offer(), LAUNCH, timeout_s=5) as host:
        assert isinstance(host, RentedHost)
        assert host.ssh_host == "1.2.3.4" and host.ssh_port == 40022
        assert host.offer.id == "NVIDIA GeForce RTX 4090"
    evs = led.events()
    assert {e["event"] for e in evs} == {"rented", "running", "destroyed"}
    d = next(e for e in evs if e["event"] == "destroyed")
    assert d["destroyed"] is True and d["verify"] == "gone"
    assert d["provider"] == "runpod" and "est_cost_usd" in d


def test_rent_raises_loudly_on_failed_terminate(mk, tmp_path):
    led = VastLedger(tmp_path / "l.jsonl")
    p = mk(FakeRunPod(delete_fail=True), ledger=led)
    with pytest.raises(LeakRisk, match="LEAK RISK"):
        with p.rent(_offer(), LAUNCH, timeout_s=5):
            pass
    d = next(e for e in led.events() if e["event"] == "destroyed")
    assert d["destroyed"] is False


def test_rent_raises_when_pod_still_present(mk):
    # terminate "succeeds" but the pod never leaves the list -> confirmed leak
    p = mk(FakeRunPod(delete_noop=True))
    with pytest.raises(LeakRisk, match="LEAK RISK"):
        with p.rent(_offer(), LAUNCH, timeout_s=5):
            pass


def test_rent_already_gone_404_terminate_is_success_not_leak(mk, tmp_path):
    """A 404 on terminate means the pod is already gone -- success, not a leak: no
    retry against a nonexistent pod, no spurious LeakRisk (CM review #48)."""
    led = VastLedger(tmp_path / "l.jsonl")
    p = mk(FakeRunPod(delete_404=True), ledger=led)
    with p.rent(_offer(), LAUNCH, timeout_s=5):     # must NOT raise LeakRisk
        pass
    d = next(e for e in led.events() if e["event"] == "destroyed")
    assert d["destroyed"] is True and d["verify"] == "gone"


def test_wait_running_bails_on_dead_status(mk):
    p = mk(FakeRunPod(status="TERMINATED"))
    with pytest.raises(HostProbeFailed):
        with p.rent(_offer(), LAUNCH, timeout_s=5):
            pass


def test_rent_tears_down_on_dead_status(mk, tmp_path):
    """A failed-to-come-up pod is still terminated (no leak) and logged."""
    led = VastLedger(tmp_path / "l.jsonl")
    p = mk(FakeRunPod(status="FAILED"), ledger=led)
    with pytest.raises(HostProbeFailed):
        with p.rent(_offer(), LAUNCH, timeout_s=5):
            pass
    d = next(e for e in led.events() if e["event"] == "destroyed")
    assert d["outcome"] == "host_failed" and d["verify"] == "gone"


def test_create_retries_transient_capacity(monkeypatch):
    """A 'does not have the resources / try a different machine' 500 is transient
    capacity -> retried (each attempt may land on a different machine)."""
    monkeypatch.setattr(runpod.time, "sleep", lambda s: None)
    calls = {"n": 0}

    def fake_req(method, url, key, payload=None, timeout=30):
        if method == "POST" and url.endswith("/pods"):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RunPodError("POST .../pods -> HTTP 500: create pod: This "
                                  "machine does not have the resources to deploy "
                                  "your pod. Please try a different machine")
            return {"id": "pod9"}
        return {}
    monkeypatch.setattr(runpod, "_req", fake_req)
    p = RunPodProvider(api_key="k")
    assert p.create(_offer(), LAUNCH) == "pod9" and calls["n"] == 3


def test_create_raises_immediately_on_non_capacity_error(monkeypatch):
    monkeypatch.setattr(runpod.time, "sleep", lambda s: None)

    def fake_req(method, url, key, payload=None, timeout=30):
        raise RunPodError("POST .../pods -> HTTP 401: unauthorized")
    monkeypatch.setattr(runpod, "_req", fake_req)
    with pytest.raises(RunPodError, match="401"):
        RunPodProvider(api_key="k").create(_offer(), LAUNCH)


def test_read_key_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    kf = tmp_path / "rp_key"
    kf.write_text("rpa_filekey\n")
    monkeypatch.setattr(runpod, "_KEY_PATHS", (str(kf),))
    assert runpod._read_key() == "rpa_filekey"


# -- the reap / control-plane contract (issue #45) ----------------------------
def test_list_instances_returns_pods_with_normalized_label(mk):
    f = FakeRunPod()
    f.pods = {"podA": {"id": "podA", "desiredStatus": "RUNNING",
                       "costPerHr": 0.22, "name": "farm-x"},
              "podB": {"id": "podB", "desiredStatus": "EXITED", "name": "farm-y"}}
    insts = mk(f).list_instances()
    by = {i.id: i for i in insts}
    assert set(by) == {"podA", "podB"}
    assert by["podA"].dph == 0.22 and by["podA"].status == "RUNNING"
    assert by["podA"].raw["label"] == "farm-x"          # name normalized to label
    assert by["podB"].dph == 0.0                          # missing cost -> 0


def test_destroy_aliases_terminate(mk):
    f = FakeRunPod()
    f.pods = {"podA": f._pod("podA")}
    mk(f).destroy("podA")
    assert "podA" not in f.pods                          # gone


def test_dead_reason_flags_dead_status_only(mk):
    f = FakeRunPod()
    f.pods = {"p": f._pod("p")}                           # RUNNING
    p = mk(f)
    assert p.dead_reason("p") is None
    f.pods["p"]["desiredStatus"] = "FAILED"
    assert "FAILED" in p.dead_reason("p")


def test_create_stamps_label_as_pod_name(mk):
    f = FakeRunPod()
    p = mk(f)
    offs = p.offers(HostSpec(gpu_name="RTX_3090", max_dph=0.30, num_gpus=1))
    p.create(offs[0], LaunchSpec(image="img", onstart="x", label="my-farm"))
    assert f.last_create["name"] == "my-farm"            # reap --label attribution
