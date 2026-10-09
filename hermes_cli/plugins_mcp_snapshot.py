"""Non-blocking view of plugin-provided (portable) MCP servers for status surfaces.

``discover_plugins()`` takes the plugin manager's discovery lock even when discovery already
ran, so a status read made while another thread holds it (the startup background scan, a
forced rescan, a subagent build in another session) waits for that whole sweep. UI status
RPCs poll, and a parked status RPC freezes the desktop for the sweep's duration. Status
readers pass ``wait=False`` and get the last complete view instead.
"""

from __future__ import annotations

import threading
import weakref
from typing import Any

# Last complete view per plugin manager: ``(servers, owning plugin per server)``. Keyed weakly by
# the manager so a new manager (profile switch, test reset) never sees another manager's view.
_last: "weakref.WeakKeyDictionary[Any, tuple[dict[str, dict[str, Any]], dict[str, str]]]" = (
    weakref.WeakKeyDictionary()
)
_last_lock = threading.Lock()


def plugin_discovery_busy(manager: Any = None) -> bool:
    """True while plugin discovery runs on ANOTHER thread. Never blocks; the calling thread's
    own re-entrant hold of the discovery lock does not count."""
    from hermes_cli import plugins

    t = plugins._background_discovery_thread
    if t is not None and t.is_alive() and t is not threading.current_thread():
        return True
    lock = getattr(manager or plugins.get_plugin_manager(), "_discovery_lock", None)
    if lock is None:
        return False
    if not lock.acquire(blocking=False):
        return True
    lock.release()
    return False


def _copy(servers: dict, owners: dict) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    return {n: dict(c) for n, c in servers.items() if isinstance(c, dict)}, dict(owners)


def portable_mcp_snapshot(*, wait: bool = True) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """``(portable MCP server configs, owning plugin per server)`` for the active home, as copies.

    ``wait=True`` discovers first (joining any in-flight sweep), like every other consumer.
    ``wait=False`` never waits on a discovery running on another thread: it returns the last
    complete snapshot for this manager (empty if there is none yet).
    """
    from hermes_cli import plugins

    manager = plugins.get_plugin_manager()
    if not wait and plugin_discovery_busy(manager):
        with _last_lock:
            servers, owners = _last.get(manager, ({}, {}))
        return _copy(servers, owners)
    plugins.discover_plugins()
    servers = manager.get_portable_mcp_servers()
    owners_fn = getattr(manager, "get_portable_mcp_server_plugins", None)
    owners = owners_fn() if callable(owners_fn) else {}
    snapshot = _copy(servers, owners)
    try:
        with _last_lock:
            _last[manager] = snapshot
    except TypeError:  # manager not weak-referenceable (test doubles); nothing to remember
        pass
    return _copy(*snapshot)
