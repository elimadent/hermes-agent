"""MCP status reads must not park a UI RPC behind another session's plugin discovery.

``session.info`` (through ``get_mcp_status``) and ``mcp.servers.status`` used to call
``discover_plugins()``, which takes the plugin manager's discovery lock even when discovery has
already run. While another session held that lock (a subagent build, a forced rescan) every
status RPC waited for the whole sweep, and the desktop froze for seconds. Status reads now serve
the last complete plugin view instead.
"""

from __future__ import annotations

import threading

import pytest

from hermes_cli import plugins as plugins_mod
from tui_gateway import server

_PORTABLE = {"demo__worker": {"command": "demo-bin"}}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    root = tmp_path / "hermes_home"
    root.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(root))
    plugins_mod._reset_plugin_managers_for_tests()
    mgr = plugins_mod.get_plugin_manager()
    monkeypatch.setattr(mgr, "_discovered", True)
    monkeypatch.setattr(mgr, "get_portable_mcp_servers", lambda: {k: dict(v) for k, v in _PORTABLE.items()})
    monkeypatch.setattr(mgr, "get_portable_mcp_server_plugins", lambda: {"demo__worker": "demo"})
    yield mgr
    plugins_mod._reset_plugin_managers_for_tests()


class _HeldDiscovery:
    """Hold the discovery lock on another thread, as a running sweep does."""

    def __init__(self, mgr):
        self._mgr, self._held, self._release = mgr, threading.Event(), threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        with self._mgr._discovery_lock:
            self._held.set()
            self._release.wait(timeout=30)

    def __enter__(self):
        self._thread.start()
        assert self._held.wait(timeout=5)
        return self

    def __exit__(self, *exc):
        self._release.set()
        self._thread.join(timeout=5)


def _within(seconds, fn):
    out: dict = {}
    worker = threading.Thread(target=lambda: out.update(value=fn()), daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    assert not worker.is_alive(), f"status read blocked > {seconds}s behind plugin discovery"
    return out["value"]


def _status_rpc():
    resp = server._methods["mcp.servers.status"](1, {})
    assert "error" not in resp, resp.get("error")
    return resp["result"]["servers"]


def test_mcp_servers_status_serves_last_plugin_view_while_discovery_runs(manager):
    assert [(s["name"], s["source"]) for s in _status_rpc()] == [("demo__worker", "plugin")]
    with _HeldDiscovery(manager):
        servers = _within(2.0, _status_rpc)
    assert [(s["name"], s["source"], s["plugin"]) for s in servers] == [("demo__worker", "plugin", "demo")]


def test_session_info_mcp_status_does_not_wait_for_discovery(manager):
    from tools.mcp_tool_discovery import get_mcp_status

    assert [s["name"] for s in get_mcp_status()] == ["demo__worker"]
    with _HeldDiscovery(manager):
        servers = _within(2.0, lambda: get_mcp_status(wait_for_plugins=False))
    assert [s["name"] for s in servers] == ["demo__worker"]


def test_default_reads_still_join_the_sweep(manager):
    from tools.mcp_tool_discovery import get_mcp_status

    with _HeldDiscovery(manager) as held:
        out: dict = {}
        worker = threading.Thread(target=lambda: out.update(value=get_mcp_status()), daemon=True)
        worker.start()
        worker.join(timeout=0.3)
        assert worker.is_alive(), "callers that need a complete registry must still wait"
        held._release.set()
        worker.join(timeout=5)
    assert [s["name"] for s in out["value"]] == ["demo__worker"]
