"""VastClient lifecycle tests with a mocked HTTP layer — no network, no spend.

All HTTP goes through `vast._req`, so a scriptable fake covers offers/create/
status/list/destroy and the cost-safety `rent()` teardown logic (the part that
must surface a leak loudly rather than pass silently).
"""

import pytest

import run_farm.vast as vast
from run_farm.vast import (
    HostProbeFailed,
    HostSpec,
    LaunchSpec,
    LeakRisk,
    Offer,
    RentedHost,
    VastClient,
    VastError,
    VastLedger,
)

OFFER = Offer(id="111", dph=0.15, gpu_name="RTX 3090", num_gpus=1, reliability=0.99,
              inet_down_mbps=900, cuda_max=12.5, geolocation=", CA", provider="vast")
LAUNCH = LaunchSpec(image="img", onstart="cmd", disk_gb=24)


class FakeVast:
    """Scriptable replacement for vast._req, routing by (method, url)."""

    def __init__(self, *, start_status="running", status_msg="",
                 destroy_fail=False, destroy_noop=False, destroy_404=False,
                 ssh_host="1.2.3.4", ssh_port=22000):
        self.inst = {}                       # id -> instance dict
        self.start_status, self.status_msg = start_status, status_msg
        self.destroy_fail, self.destroy_noop = destroy_fail, destroy_noop
        self.destroy_404 = destroy_404       # box gone, but DELETE returns HTTP 404
        self.ssh_host, self.ssh_port = ssh_host, ssh_port
        self.offers = [
            dict(id=111, dph_total=0.15, gpu_name="RTX 3090", num_gpus=1,
                 reliability2=0.99, inet_down=900, cuda_max_good=12.5, geolocation=", CA"),
            dict(id=112, dph_total=0.60, gpu_name="RTX 3090", num_gpus=1,
                 reliability2=0.99, inet_down=900, cuda_max_good=12.5, geolocation=", CA"),
        ]

    def __call__(self, method, url, key, payload=None, timeout=30, **kw):
        if method == "POST" and "/bundles/" in url:
            return {"offers": self.offers}
        if method == "PUT" and "/asks/" in url:
            self.inst[9001] = {"actual_status": self.start_status,
                               "dph_total": 0.15, "status_msg": self.status_msg,
                               "ssh_host": self.ssh_host, "ssh_port": self.ssh_port}
            return {"new_contract": 9001}
        if method == "GET" and "/api/v1/instances/" in url:
            return {"instances": [{"id": i, **d} for i, d in self.inst.items()]}
        if method == "GET" and "/api/v0/instances/" in url:
            iid = int(url.split("/instances/")[1].split("/")[0])
            return {"instances": self.inst.get(iid, {})}
        if method == "DELETE" and "/instances/" in url:
            if self.destroy_404:             # already gone: API 404s, box IS absent
                iid = int(url.split("/instances/")[1].split("/")[0])
                self.inst.pop(iid, None)
                err = VastError("DELETE .../ -> HTTP 404: instance not found")
                err.code = 404
                raise err
            if self.destroy_fail:
                raise VastError("destroy boom")
            if not self.destroy_noop:
                iid = int(url.split("/instances/")[1].split("/")[0])
                self.inst.pop(iid, None)
            return {}
        return {}


@pytest.fixture
def mk(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "testkey")
    monkeypatch.setattr(vast.time, "sleep", lambda s: None)   # don't really wait

    def make(fake, ledger=None):
        monkeypatch.setattr(vast, "_req", fake)
        return VastClient(ledger=ledger)
    return make


def test_cheapest_offer_filters_by_price(mk):
    o = mk(FakeVast()).cheapest_offer(
        HostSpec(gpu_name="RTX_3090", max_dph=0.30, min_reliability=0.9,
                 min_inet_mbps=100, min_cuda=12.0))
    assert o.id == "111" and o.dph == 0.15      # 112 ($0.60) filtered out
    assert o.provider == "vast"                  # adapter stamps the offer


def test_offers_min_gpu_frac_gate(mk):
    """min_gpu_frac>0 adds the dedicated-machine gpu_frac gate to the /bundles/
    query (anti-contention); the default 0.0 omits it (backward-compatible)."""
    captured = {}
    base = FakeVast()

    def capture(method, url, key, payload=None, timeout=30, **kw):
        if method == "POST" and "/bundles/" in url:
            captured["q"] = payload
        return base(method, url, key, payload, timeout, **kw)

    prov = mk(capture)
    base_spec = dict(gpu_name="RTX_3090", max_dph=0.30, min_reliability=0.9,
                     min_inet_mbps=100, min_cuda=12.0)
    prov.offers(HostSpec(**base_spec, min_gpu_frac=1.0))
    assert captured["q"].get("gpu_frac") == {"gte": 1.0}   # gated -> dedicated only
    captured.clear()
    prov.offers(HostSpec(**base_spec))                     # default 0.0
    assert "gpu_frac" not in captured["q"]                 # no gate -> unchanged query


def test_rent_happy_path_destroys_and_verifies_gone(mk, tmp_path):
    led = VastLedger(tmp_path / "l.jsonl")
    c = mk(FakeVast(start_status="running"), ledger=led)
    with c.rent(OFFER, LAUNCH, timeout_s=5) as host:
        assert isinstance(host, RentedHost) and host.id == "9001"
        assert host.ssh_host == "1.2.3.4" and host.ssh_port == 22000   # reachable
        assert host.offer is OFFER               # cost/geo stay attached
    evs = led.events()
    assert {e["event"] for e in evs} == {"rented", "running", "destroyed"}
    assert all(e["provider"] == "vast" for e in evs)   # cross-provider attribution
    d = next(e for e in evs if e["event"] == "destroyed")
    assert d["destroyed"] is True and d["verify"] == "gone"
    assert "est_cost_usd" in d


def test_rent_missing_ssh_coords_fails_over_and_destroys(mk, tmp_path):
    """A 'running' instance with no SSH coordinates is unusable -> HostProbeFailed
    (so the executor fails over), and teardown still fires + verifies gone."""
    led = VastLedger(tmp_path / "l.jsonl")
    c = mk(FakeVast(start_status="running", ssh_host="", ssh_port=0), ledger=led)
    with pytest.raises(HostProbeFailed):
        with c.rent(OFFER, LAUNCH, timeout_s=5):
            pass
    evs = led.events()
    d = next(e for e in evs if e["event"] == "destroyed")
    assert d["outcome"] == "host_failed" and d["verify"] == "gone"


def test_create_rejects_non_int_offer_id_as_vast_error(mk):
    """A foreign Provider's Offer id (non-int) yields a clear VastError, not a
    bare ValueError from int()."""
    c = mk(FakeVast())
    with pytest.raises(VastError, match="not a Vast ask id"):
        c.create("NVIDIA GeForce RTX 4090", image="img", onstart_cmd="echo")


def test_rent_raises_loudly_on_failed_destroy(mk, tmp_path):
    led = VastLedger(tmp_path / "l.jsonl")
    c = mk(FakeVast(start_status="running", destroy_fail=True), ledger=led)
    with pytest.raises(LeakRisk, match="LEAK RISK"):
        with c.rent(OFFER, LAUNCH, timeout_s=5):
            pass
    d = next(e for e in led.events() if e["event"] == "destroyed")
    assert d["destroyed"] is False             # recorded, not silently passed


def test_rent_raises_when_instance_still_present(mk):
    # destroy "succeeds" but the instance never leaves the list -> confirmed leak
    c = mk(FakeVast(start_status="running", destroy_noop=True))
    with pytest.raises(LeakRisk, match="LEAK RISK"):
        with c.rent(OFFER, LAUNCH, timeout_s=5):
            pass


def test_rent_already_gone_404_destroy_is_success_not_leak(mk, tmp_path):
    """A 404 on destroy means the box is already gone -- the DESIRED state. It must
    count as torn down (no retry against a nonexistent instance, no spurious
    LeakRisk), with verify confirming gone (CM review #48)."""
    led = VastLedger(tmp_path / "l.jsonl")
    c = mk(FakeVast(start_status="running", destroy_404=True), ledger=led)
    with c.rent(OFFER, LAUNCH, timeout_s=5):        # must NOT raise LeakRisk
        pass
    d = next(e for e in led.events() if e["event"] == "destroyed")
    assert d["destroyed"] is True and d["verify"] == "gone"


def test_wait_running_bails_on_bad_host(mk):
    fake = FakeVast(start_status="loading",
                    status_msg="failed to resolve auth.docker.io")
    c = mk(fake)
    iid = c.create(OFFER.id, image="img", onstart_cmd="cmd")
    with pytest.raises(HostProbeFailed):
        c.wait_running(iid, timeout_s=5, poll_s=0)


def test_list_instances_uses_v1(mk):
    fake = FakeVast()
    c = mk(fake)
    fake.inst[42] = {"actual_status": "running", "dph_total": 0.2}
    assert [i.id for i in c.list_instances()] == [42]


def test_read_key_from_file(tmp_path, monkeypatch):
    monkeypatch.delenv("VAST_API_KEY", raising=False)
    kf = tmp_path / "vast_key"
    kf.write_text("filekey\n")
    monkeypatch.setattr(vast, "_KEY_PATHS", (str(kf),))
    assert vast._read_key() == "filekey"


def test_req_raises_vasterror_on_http_error(monkeypatch):
    import io
    import urllib.error

    def boom(req, timeout=30):
        raise urllib.error.HTTPError(req.full_url, 410, "Gone", {}, io.BytesIO(b"dead"))
    monkeypatch.setattr(vast.urllib.request, "urlopen", boom)
    with pytest.raises(VastError, match="410"):
        vast._req("GET", "https://x/api/v0/instances/", "k")


def test_logs_polls_result_url(monkeypatch):
    monkeypatch.setenv("VAST_API_KEY", "k")
    monkeypatch.setattr(vast, "_req",
                        lambda *a, **k: {"result_url": "https://s3/log.txt"})

    class Resp:
        status = 200
        def read(self): return b"onstart output\n=== DONE ==="
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(vast.urllib.request, "urlopen",
                        lambda url, timeout=20: Resp())
    assert "DONE" in VastClient().logs(123)


def test_rent_failed_over_host_logs_outcome(mk, tmp_path):
    led = VastLedger(tmp_path / "l.jsonl")
    c = mk(FakeVast(start_status="loading", status_msg="failed to resolve x"),
           ledger=led)
    with pytest.raises(HostProbeFailed):
        with c.rent(OFFER, LAUNCH, timeout_s=5):
            pass
    d = next(e for e in led.events() if e["event"] == "destroyed")
    assert d["outcome"] == "host_failed" and d["verify"] == "gone"


# -- transient-fault retry in _req (#23) -------------------------------------
# These exercise the REAL _req (not the FakeVast that replaces it), against a
# scripted urlopen, so the self-healing retry/backoff is actually covered.
import io as _io                                              # noqa: E402
import urllib.error as _ue                                   # noqa: E402


class _Resp:
    def __init__(self, body=b"{}"):
        self._body = body
    def read(self):
        return self._body
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _seq_urlopen(behaviors):
    """A fake urlopen consuming `behaviors` one call at a time; each item is an
    Exception to raise or `bytes` to return as a response body."""
    it = iter(behaviors)

    def fake(req, timeout=30):
        b = next(it)
        if isinstance(b, BaseException):
            raise b
        return _Resp(b)
    return fake


def _dns_error():
    """The EAI_AGAIN a saturated resolver throws -- a TRANSIENT pre-send failure."""
    return _ue.URLError(vast.socket.gaierror(
        vast.socket.EAI_AGAIN, "Temporary failure in name resolution"))


def _dns_terminal():
    """EAI_NONAME -- the name genuinely doesn't resolve (terminal, not retried)."""
    return _ue.URLError(vast.socket.gaierror(
        vast.socket.EAI_NONAME, "Name or service not known"))


@pytest.fixture
def no_sleep(monkeypatch):
    """Record (don't perform) backoff sleeps so retries are instant + countable."""
    calls = []
    monkeypatch.setattr(vast.time, "sleep", lambda s: calls.append(s))
    return calls


def test_req_retries_transient_dns_then_succeeds(monkeypatch, no_sleep):
    monkeypatch.setattr(vast.urllib.request, "urlopen",
                        _seq_urlopen([_dns_error(), _dns_error(), b'{"ok": 1}']))
    assert vast._req("GET", "https://x/api/v1/instances/", "k") == {"ok": 1}
    assert len(no_sleep) == 2                        # backed off before each retry
    assert no_sleep[1] > no_sleep[0]                 # exponential


def test_rent_terminal_auth_error_propagates_as_vasterror(mk, monkeypatch):
    """A 401/403 from create is a terminal config error (bad key) -- it must
    surface, not be disguised as an offer race that burns the whole pool."""
    from run_farm.vast import RentUnavailable  # noqa: F401
    c = mk(FakeVast())
    err = VastError("PUT /asks -> HTTP 403: forbidden"); err.code = 403

    def boom(*a, **k):
        raise err
    monkeypatch.setattr(c, "create", boom)
    with pytest.raises(VastError):
        with c.rent(OFFER, LAUNCH, timeout_s=5):
            pass


def test_rent_offer_race_becomes_rentunavailable(mk, monkeypatch):
    """A non-auth create failure (e.g. 404 ask-gone) created no instance -> a
    recoverable RentUnavailable the executor fails over on."""
    from run_farm.vast import RentUnavailable
    c = mk(FakeVast())
    err = VastError("PUT /asks -> HTTP 404: no_such_ask"); err.code = 404

    def boom(*a, **k):
        raise err
    monkeypatch.setattr(c, "create", boom)
    with pytest.raises(RentUnavailable):
        with c.rent(OFFER, LAUNCH, timeout_s=5):
            pass


def test_req_terminal_dns_is_not_retried(monkeypatch, no_sleep):
    """A name that genuinely doesn't resolve (EAI_NONAME) must fail immediately --
    retrying it just burns backoff and hides a misconfiguration."""
    monkeypatch.setattr(vast.urllib.request, "urlopen",
                        _seq_urlopen([_dns_terminal()]))
    with pytest.raises(VastError):
        vast._req("GET", "https://x/api/v1/instances/", "k")
    assert no_sleep == []                              # no retry, no backoff


def test_req_exhausts_retries_then_raises_vasterror(monkeypatch, no_sleep):
    monkeypatch.setattr(vast.urllib.request, "urlopen",
                        _seq_urlopen([_dns_error()] * 5))
    with pytest.raises(VastError):
        vast._req("GET", "https://x/api/v1/instances/", "k", tries=5)
    assert len(no_sleep) == 4                         # 5 tries -> 4 backoffs


def test_req_5xx_retries_but_4xx_is_terminal(monkeypatch, no_sleep):
    e503 = _ue.HTTPError("u", 503, "busy", {}, _io.BytesIO(b"busy"))
    e400 = _ue.HTTPError("u", 400, "bad", {}, _io.BytesIO(b"bad request"))
    monkeypatch.setattr(vast.urllib.request, "urlopen", _seq_urlopen([e503, e400]))
    with pytest.raises(VastError, match="400"):
        vast._req("GET", "https://x/api/v1/instances/", "k")
    assert len(no_sleep) == 1                         # one backoff (after the 503)


def test_req_nonidempotent_does_not_retry_post_connect_fault(monkeypatch, no_sleep):
    """create (idempotent=False) must NOT retry a connection reset: the request
    may have already rented a GPU, so a retry would double-rent (a leak)."""
    reset = _ue.URLError(ConnectionResetError("reset"))
    monkeypatch.setattr(vast.urllib.request, "urlopen", _seq_urlopen([reset]))
    with pytest.raises(VastError):
        vast._req("PUT", "https://x/api/v0/asks/1/", "k", {"a": 1}, idempotent=False)
    assert no_sleep == []                             # failed immediately, no retry


def test_req_nonidempotent_still_retries_dns(monkeypatch, no_sleep):
    """A DNS failure is pre-send (server never saw it), so even a create safely
    retries it -- the exact fault that killed the 2026-06-15 farm."""
    monkeypatch.setattr(vast.urllib.request, "urlopen",
                        _seq_urlopen([_dns_error(), b'{"new_contract": 7}']))
    out = vast._req("PUT", "https://x/api/v0/asks/1/", "k", {"a": 1}, idempotent=False)
    assert out == {"new_contract": 7}
    assert len(no_sleep) == 1


def test_req_retries_bare_timeout_then_raises_with_reason(monkeypatch, no_sleep):
    """A bare TimeoutError (not wrapped in URLError) is transient for an
    idempotent call; on exhaustion it surfaces as VastError naming the cause."""
    monkeypatch.setattr(vast.urllib.request, "urlopen",
                        _seq_urlopen([TimeoutError("read timed out")] * 3))
    with pytest.raises(VastError, match="TimeoutError"):
        vast._req("GET", "https://x/api/v1/instances/", "k", tries=3)
    assert len(no_sleep) == 2                          # 3 tries -> 2 backoffs


def test_launch_label_is_passed_to_create(mk):
    """LaunchSpec.label stamps the rented instance so reap can scope by label."""
    fake = FakeVast(start_status="running")
    seen = {}
    orig = fake.__call__

    def spy(method, url, key, payload=None, timeout=30, **kw):
        if method == "PUT" and "/asks/" in url:
            seen["label"] = payload.get("label")
        return orig(method, url, key, payload, timeout, **kw)
    c = mk(spy)
    with c.rent(OFFER, LaunchSpec(image="img", onstart="cmd", label="farm-xyz")):
        pass
    assert seen["label"] == "farm-xyz"
