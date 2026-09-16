"""A session that is WORKING must read as working in the session list.

The bug this locks down (reported from the iOS app, 2026-09-16): a session
sitting on ``Mulling 15m 31s`` / ``Waiting on grok-4.6 (call 30)`` — plainly
alive — showed no working marker in the session list.

``is_active`` on a session row was five copy-pasted instances of::

    s.get("ended_at") is None
    and (now - s.get("last_active", s.get("started_at", 0))) < 300

That is a RECENCY test, not a liveness test, and ``last_active`` is
``MAX(messages.timestamp)`` (``hermes_state.list_sessions_rich``) — it only
moves when a row is persisted. So the indicator answers "was this session
touched in the last five minutes?" while every UI that renders it is asking
"is the agent working this session?". The two answers diverge in BOTH
directions, and the width of the gap is one turn:

* **False negative** (the reported bug): a turn that spends minutes inside a
  single model call, a slow tool, or deep reasoning writes no rows, so after
  300s the session goes dark while it is demonstrably working.
* **False positive**: a session that finished four minutes ago still reads
  as working.

The gateway already knows the truth. ``SessionTurnLeaseRegistry`` holds a
lease per RESOLVED session_id for the whole ``[load history -> run -> flush]``
region, which is exactly the life of a turn — and it is keyed by the id REST
lists, unlike ``_running_agents`` (routing key, many-to-one onto session_id).
The gateway now publishes the held ids into ``gateway_state.json``, and the
REST layer reads them.

The recency window is KEPT as an OR on purpose. An older gateway, a stopped
one, or an unreadable state file all yield an empty set, and every row then
falls back to byte-identical previous behaviour. This change can restore a
marker that was wrongly missing; it can never remove one that was there.
"""
from __future__ import annotations

import ast
import asyncio
import io
import json
import types
from pathlib import Path

import pytest

from gateway import status as gateway_status
from gateway.turn_lease import SessionTurnLeaseRegistry

REPO_ROOT = Path(__file__).resolve().parents[2]
NOW = 1_000_000.0


# ---------------------------------------------------------------------------
# The gateway's answer to "is a turn in flight on this session?"
# ---------------------------------------------------------------------------

def test_held_session_ids_tracks_real_turns():
    async def go():
        reg = SessionTurnLeaseRegistry()
        assert reg.held_session_ids() == []

        a = await reg.acquire("sess_A", owner_key="key1", generation=1)
        assert reg.held_session_ids() == ["sess_A"]

        b = await reg.acquire("sess_B", owner_key="key2", generation=1)
        assert sorted(reg.held_session_ids()) == ["sess_A", "sess_B"]

        reg.release(a)
        assert reg.held_session_ids() == ["sess_B"]

        # THE POINT: the lease is held for the whole turn, so a turn that has
        # written nothing for fifteen minutes is still reported.
        assert "sess_B" in reg.held_session_ids()

        reg.release(b)
        assert reg.held_session_ids() == []

    asyncio.run(go())


def test_mid_turn_compression_reports_both_ids():
    """A rebind aliases one lease onto two ids; reporting both is honest.

    Over-reporting for the width of the rotation window says "something is
    running on this lineage", which is true. Under-reporting would blank the
    marker on the row the user is actually looking at.
    """
    async def go():
        reg = SessionTurnLeaseRegistry()
        tok = await reg.acquire("sess_B", owner_key="k", generation=1)
        assert reg.rebind(tok, "sess_B_tip") is True
        assert sorted(reg.held_session_ids()) == ["sess_B", "sess_B_tip"]
        reg.release(tok)
        assert reg.held_session_ids() == []

    asyncio.run(go())


def test_degraded_waiter_does_not_claim_the_session():
    """A timed-out (fail-open) turn holds nothing, so it must not be reported."""
    async def go():
        reg = SessionTurnLeaseRegistry()
        holder = await reg.acquire("sess_C", owner_key="k", generation=1)
        waiter = await reg.acquire(
            "sess_C", owner_key="k2", generation=2, timeout=0.05
        )
        assert waiter.degraded is True
        assert reg.held_session_ids() == ["sess_C"]
        reg.release(holder)

    asyncio.run(go())


# ---------------------------------------------------------------------------
# Publishing it through gateway_state.json
# ---------------------------------------------------------------------------

def test_write_runtime_status_merges_session_ids(tmp_path, monkeypatch):
    path = tmp_path / "gateway_state.json"
    monkeypatch.setattr(gateway_status, "_get_runtime_status_path", lambda: path)

    gateway_status.write_runtime_status(
        gateway_state="running", active_session_ids=["b", "a", "a"]
    )
    rec = json.loads(path.read_text())
    # Sorted + de-duplicated so a dict-order-only change never churns the file.
    assert rec["active_session_ids"] == ["a", "b"]

    # A write that omits the field must PRESERVE it — the turn-boundary writes
    # pass only what they own. Losing the _UNSET sentinel here would also make
    # every such call raise, and both callers swallow exceptions, so the state
    # file would silently stop updating altogether.
    gateway_status.write_runtime_status(active_agents=3)
    rec = json.loads(path.read_text())
    assert rec["active_session_ids"] == ["a", "b"]
    assert rec["gateway_state"] == "running"

    # An empty set CLEARS the marker. This is the restart path: no turn
    # survives a gateway restart, so the lifecycle write on the way to
    # "running" is what clears a marker left by a hard kill mid-turn.
    gateway_status.write_runtime_status(active_session_ids=[])
    rec = json.loads(path.read_text())
    assert rec["active_session_ids"] == []
    assert rec["gateway_state"] == "running"


# ---------------------------------------------------------------------------
# The REST layer's rule, run from the SHIPPED source
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def session_is_active():
    """Extract and execute the real ``_session_is_active``.

    Importing ``hermes_cli.web_server`` drags in the whole dashboard; lifting
    the function out of the shipped file keeps this test honest (it cannot
    drift from what ships) without that cost.
    """
    src = io.open(REPO_ROOT / "hermes_cli" / "web_server.py", encoding="utf-8").read()
    fn = next(
        node
        for node in ast.parse(src).body
        if isinstance(node, ast.FunctionDef) and node.name == "_session_is_active"
    )
    mod = types.ModuleType("_extracted_web_server")
    mod.__dict__.update({"Dict": dict, "Any": object})
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "web_server.py", "exec"),
         mod.__dict__)
    return mod._session_is_active


def _row(**kw):
    row = {"id": "s1", "ended_at": None, "last_active": NOW - 10}
    row.update(kw)
    return row


def test_silent_long_turn_reads_as_working(session_is_active):
    """The reported bug, end to end.

    Twenty minutes since the last transcript row, so recency alone says idle —
    but the gateway is holding this session's turn lease, so it is working.
    """
    long_turn = _row(last_active=NOW - 1200)
    assert session_is_active(long_turn, NOW, frozenset({"s1"})) is True
    # And the same row WITHOUT the gateway signal is the old (wrong) answer,
    # which is what pins the fix to the new signal rather than to the clock.
    assert session_is_active(long_turn, NOW, frozenset()) is False


def test_empty_gateway_set_is_byte_identical_to_the_old_rule(session_is_active):
    """Zero-regression guard: no gateway signal -> exactly previous behaviour."""

    def old_rule(row, now):
        if row.get("ended_at") is not None:
            return False
        return (now - row.get("last_active", row.get("started_at", 0))) < 300

    rows = [
        _row(last_active=NOW - age, ended_at=ended)
        for age in (0, 1, 299, 300, 301, 5000)
        for ended in (None, NOW - 1)
    ]
    rows.append({"id": "x", "ended_at": None, "started_at": NOW - 5})
    rows.append({"id": "x", "ended_at": None, "started_at": NOW - 9999})

    for row in rows:
        assert session_is_active(row, NOW, frozenset()) == old_rule(row, NOW), row


def test_ended_session_is_never_working(session_is_active):
    assert session_is_active(_row(ended_at=NOW - 1), NOW, frozenset({"s1"})) is False


def test_compression_projected_row_matches_on_lineage_root(session_is_active):
    """The list row's id is projected to the tip; the lease may hold the root."""
    row = _row(id="tip", _lineage_root_id="orig")
    assert session_is_active(row, NOW, frozenset({"orig"})) is True


def test_another_sessions_turn_does_not_light_this_row(session_is_active):
    assert session_is_active(_row(last_active=NOW - 9999), NOW,
                             frozenset({"other"})) is False


def test_unusable_timestamps_never_raise(session_is_active):
    assert session_is_active(_row(last_active="not-a-number"), NOW,
                             frozenset()) is False
    assert session_is_active(
        {"id": "s", "ended_at": None, "last_active": None, "started_at": NOW - 5},
        NOW, frozenset(),
    ) is True


def test_every_list_surface_uses_the_shared_rule():
    """No surface may keep its own copy of the recency expression.

    There were five identical inline copies plus a sixth in the /api/status
    count; they must agree about what "working" means or the dashboard and the
    list will contradict each other.
    """
    src = io.open(REPO_ROOT / "hermes_cli" / "web_server.py", encoding="utf-8").read()
    assert 'and (now - s.get("last_active"' not in src
    assert src.count("_session_is_active(") >= 7   # 1 definition + 6 call sites
    assert src.count("_running_session_ids(") >= 7
