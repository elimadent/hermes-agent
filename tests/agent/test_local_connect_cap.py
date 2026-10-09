"""Connect-phase cap for non-loopback, plain-http local model hosts (agent.process_bootstrap).

An unanswered TCP handshake to a powered-off LAN/Tailscale model host used to stall each request
for the OS retry window (~21 s on Windows) before the fallback ran. Only the connect timeout is
capped; read/write/pool and loopback/https/cloud endpoints are untouched.
"""
import asyncio
import socket
import threading
import time

import httpx
import pytest

from agent import process_bootstrap as pb


@pytest.mark.parametrize("url,expected", [
    ("http://100.64.0.10:11434/v1", 5.0),          # Tailscale CGNAT
    ("http://192.168.1.115:11434/v1", 5.0),         # LAN
    ("http://10.0.0.5:8000/v1", 5.0),
    ("http://127.0.0.1:9120/v1", None),             # loopback: untouched
    ("http://localhost:11434/v1", None),
    ("http://[::1]:11434/v1", None),
    ("http://[0:0:0:0:0:0:0:1]:9120/v1", None),       # expanded IPv6 loopback
    ("http://127.5.5.5:9120/v1", None),
    ("https://100.64.0.10:11434/v1", None),          # TLS: connect timeout also covers the handshake
    ("https://api.openai.com/v1", None),            # cloud: untouched
    ("https://chatgpt.com/backend-api/codex", None),
    ("", None),
])
def test_cap_selection(url, expected, monkeypatch):
    assert pb._local_connect_cap_seconds(url) == expected


def test_cap_override_and_disable(monkeypatch):
    monkeypatch.setattr(pb, "LOCAL_CONNECT_CAP_SECONDS", 8)
    assert pb._local_connect_cap_seconds("http://100.64.0.10:11434/v1") == 8.0
    monkeypatch.setattr(pb, "LOCAL_CONNECT_CAP_SECONDS", 0)
    assert pb._local_connect_cap_seconds("http://100.64.0.10:11434/v1") is None
    client = pb.build_keepalive_http_client("http://100.64.0.10:11434/v1")
    assert client.event_hooks["request"] == []
    client.close()


def _timeout_after_hooks(client, timeout):
    req = client.build_request("GET", "http://100.64.0.10:11434/v1/models", timeout=timeout)
    for hook in client.event_hooks["request"]:
        hook(req)
    return req.extensions["timeout"]


def test_hook_lowers_connect_only(monkeypatch):
    client = pb.build_keepalive_http_client("http://100.64.0.10:11434/v1")
    t = _timeout_after_hooks(client, httpx.Timeout(connect=30.0, read=1800.0, write=1800.0, pool=30.0))
    assert t == {"connect": 5.0, "read": 1800.0, "write": 1800.0, "pool": 30.0}
    t = _timeout_after_hooks(client, httpx.Timeout(connect=2.0, read=None, write=2.0, pool=2.0))
    assert t["connect"] == 2.0 and t["read"] is None          # never raises a lower caller value
    t = _timeout_after_hooks(client, httpx.Timeout(None))
    assert t["connect"] == 5.0 and t["read"] is None
    client.close()


def test_no_hooks_for_loopback_and_cloud(monkeypatch):
    for url in ("http://127.0.0.1:9120/v1", "https://api.openai.com/v1"):
        client = pb.build_keepalive_http_client(url)
        assert client.event_hooks["request"] == []
        client.close()


def test_async_client_hook(monkeypatch):
    client = pb.build_keepalive_http_client("http://100.64.0.10:11434/v1", async_mode=True)
    req = client.build_request("GET", "http://100.64.0.10:11434/v1/models",
                               timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0))

    async def run():
        for hook in client.event_hooks["request"]:
            await hook(req)
        await client.aclose()
    asyncio.run(run())
    assert req.extensions["timeout"]["connect"] == 5.0 and req.extensions["timeout"]["read"] == 600.0


def test_served_model_capture_keeps_cap_hook(monkeypatch):
    """install_served_model_capture merges response hooks; the request hook must survive."""
    client = pb.build_keepalive_http_client("http://100.64.0.10:11434/v1")
    hooks = client.event_hooks
    client.event_hooks = {**hooks, "response": [*hooks.get("response", ()), lambda r: None]}
    assert len(client.event_hooks["request"]) == 1
    client.close()


def _lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.168.1.1", 9))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


@pytest.mark.skipif(_lan_ip() is None or not _lan_ip().startswith(("192.168.", "10.", "172.")),
                    reason="needs a private LAN address")
def test_slow_first_byte_still_answers(monkeypatch):
    """A host that accepts at once but takes longer than the cap to answer (= a model still loading)
    must still succeed: only the TCP handshake is capped."""
    monkeypatch.setattr(pb, "LOCAL_CONNECT_CAP_SECONDS", 1)
    ip = _lan_ip()
    srv = socket.socket()
    srv.bind((ip, 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        time.sleep(3)                                  # 3x the connect cap
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        conn.close()
    threading.Thread(target=serve, daemon=True).start()
    client = pb.build_keepalive_http_client(f"http://{ip}:{port}/v1")
    t0 = time.monotonic()
    r = client.get(f"http://{ip}:{port}/v1/models", timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0))
    assert r.status_code == 200 and r.text == "ok" and time.monotonic() - t0 >= 2.9
    client.close()
    srv.close()


def test_unreachable_host_fails_fast(monkeypatch):
    """An address nobody answers (SYN dropped) fails at the cap, not after Windows' ~21 s retries."""
    monkeypatch.setattr(pb, "LOCAL_CONNECT_CAP_SECONDS", 2)
    url = "http://10.255.255.1:11434/v1"
    client = pb.build_keepalive_http_client(url)
    t0 = time.monotonic()
    with pytest.raises((httpx.ConnectTimeout, httpx.ConnectError)):
        client.get(url + "/models", timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0))
    assert time.monotonic() - t0 < 6
    client.close()


def _redirect_server(location: bytes, bind_ip: str = "127.0.0.1"):
    srv = socket.socket()
    srv.bind((bind_ip, 0))
    srv.listen(1)

    def serve():
        conn, _ = srv.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 302 Found\r\nLocation: " + location + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        conn.close()
    threading.Thread(target=serve, daemon=True).start()
    return srv


def test_redirect_to_ineligible_host_restores_original_connect(monkeypatch):
    client = pb.build_keepalive_http_client("http://100.64.0.10:11434/v1")
    hook = client.event_hooks["request"][0]
    req = client.build_request("GET", "http://100.64.0.10:11434/v1/models",
                               timeout=httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0))
    hook(req)
    assert req.extensions["timeout"]["connect"] == 5.0
    # what httpx does on a redirect: same extensions, new URL
    for target in ("https://api.openai.com/v1/models", "http://127.0.0.1:9120/v1/models"):
        hop = httpx.Request("GET", target, extensions=dict(req.extensions))
        hook(hop)
        assert hop.extensions["timeout"]["connect"] == 30.0 and pb._ORIG_CONNECT_EXT not in hop.extensions
        assert hop.extensions["timeout"]["read"] == 600.0
    # an absolute cloud URL through a local-built client is not capped either
    req2 = client.build_request("GET", "https://api.openai.com/v1/models", timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0))
    hook(req2)
    assert req2.extensions["timeout"]["connect"] == 30.0
    client.close()


@pytest.mark.skipif(_lan_ip() is None or not _lan_ip().startswith(("192.168.", "10.", "172.")),
                    reason="needs a private LAN address")
def test_real_redirect_follow(monkeypatch):
    """End to end through httpx's redirect machinery: a CAPPED hop (LAN host) redirected to loopback
    must reach the loopback hop with the caller's original connect timeout restored."""
    ip = _lan_ip()
    seen = []
    final = socket.socket()
    final.bind(("127.0.0.1", 0))
    final.listen(1)
    fport = final.getsockname()[1]

    def serve_final():
        conn, _ = final.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok")
        conn.close()
    threading.Thread(target=serve_final, daemon=True).start()
    first = _redirect_server(f"http://127.0.0.1:{fport}/final".encode(), bind_ip=ip)
    base = f"http://{ip}:{first.getsockname()[1]}"
    client = pb.build_keepalive_http_client(base + "/v1")
    client.event_hooks = {**client.event_hooks,
                          "request": [*client.event_hooks["request"], lambda r: seen.append(dict(r.extensions["timeout"]))]}
    r = client.get(base + "/start", follow_redirects=True,
                   timeout=httpx.Timeout(connect=30.0, read=60.0, write=30.0, pool=30.0))
    assert r.text == "ok" and [t["connect"] for t in seen] == [5.0, 30.0]
    assert all(t["read"] == 60.0 for t in seen)
    client.close()
    first.close()
    final.close()
