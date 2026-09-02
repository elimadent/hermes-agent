"""Focused tests for API server session-control endpoints."""

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


@pytest.fixture
def auth_adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "sk-test"}))
    adapter._session_db = session_db
    return adapter


def _create_session_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_post("/api/sessions", adapter._handle_create_session)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_patch("/api/sessions/{session_id}", adapter._handle_patch_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_post("/api/sessions/{session_id}/chat", adapter._handle_session_chat)
    app.router.add_post("/api/sessions/{session_id}/chat/stream", adapter._handle_session_chat_stream)
    return app


def _copy_messages_as_compacted_generation(session_db, session_id):
    def _copy(conn):
        conn.execute(
            """
            INSERT INTO messages
                (session_id, role, content, tool_call_id, tool_calls,
                 tool_name, timestamp, active, compacted)
            SELECT session_id, role, content, tool_call_id, tool_calls,
                   tool_name, timestamp, 0, 1
            FROM messages
            WHERE session_id = ? AND active = 1
            """,
            [session_id],
        )

    session_db._execute_write(_copy)


@pytest.mark.asyncio
async def test_capabilities_advertises_session_control_surface(adapter):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/v1/capabilities")
        assert resp.status == 200
        data = await resp.json()

    features = data["features"]
    assert features["session_resources"] is True
    assert features["session_chat"] is True
    assert features["session_chat_streaming"] is True
    assert features["session_fork"] is True
    assert features["run_steer"] is True
    assert features["admin_config_rw"] is False
    assert features["memory_write_api"] is False
    assert features["skills_api"] is True
    assert features["realtime_voice"] is False
    assert data["endpoints"]["sessions"] == {"method": "GET", "path": "/api/sessions"}
    assert data["endpoints"]["session_chat_stream"] == {
        "method": "POST",
        "path": "/api/sessions/{session_id}/chat/stream",
    }
    assert data["endpoints"]["run_steer"] == {
        "method": "POST",
        "path": "/v1/runs/{run_id}/steer",
    }


@pytest.mark.asyncio
async def test_session_messages_default_to_latest_bounded_page(adapter, session_db):
    session_id = session_db.create_session("bounded-messages", "api_server")
    session_db.replace_messages(
        session_id,
        [{"role": "user", "content": f"msg {i}"} for i in range(501)],
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(f"/api/sessions/{session_id}/messages")
        assert resp.status == 200
        payload = await resp.json()

        explicit_resp = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=2&offset=1"
        )
        assert explicit_resp.status == 200
        explicit = await explicit_resp.json()

    assert payload["pagination"] == {
        "limit": 500,
        "offset": 0,
        "order": "latest",
        "returned": 500,
        # Existing callers keep the active-only projection by default.
        "include_compacted": False,
        # A full page means at least one more page exists below it.
        "has_more": True,
    }
    assert payload["data"][0]["content"] == "msg 1"
    assert payload["data"][-1]["content"] == "msg 500"
    assert [message["content"] for message in explicit["data"]] == [
        "msg 1",
        "msg 2",
    ]
    # A full page means more history exists below it; this session has 501.
    assert explicit["pagination"]["has_more"] is True


@pytest.mark.asyncio
async def test_session_messages_exact_full_final_page_does_not_claim_more(
    adapter, session_db
):
    session_id = session_db.create_session("exact-full-final-page", "api_server")
    session_db.append_messages_batch(
        session_id,
        [{"role": "user", "content": f"msg {i}"} for i in range(2)],
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(f"/api/sessions/{session_id}/messages?limit=2")
        assert resp.status == 200
        payload = await resp.json()

    assert [message["content"] for message in payload["data"]] == ["msg 0", "msg 1"]
    assert payload["pagination"]["returned"] == 2
    assert payload["pagination"]["has_more"] is False


@pytest.mark.asyncio
async def test_session_messages_one_extra_row_claims_more(adapter, session_db):
    session_id = session_db.create_session("one-extra-row", "api_server")
    session_db.append_messages_batch(
        session_id,
        [{"role": "user", "content": f"msg {i}"} for i in range(3)],
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(f"/api/sessions/{session_id}/messages?limit=2")
        assert resp.status == 200
        payload = await resp.json()

    assert [message["content"] for message in payload["data"]] == ["msg 0", "msg 1"]
    assert payload["pagination"]["returned"] == 2
    assert payload["pagination"]["has_more"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("logical_count", "expected_contents", "expected_has_more"),
    [(2, ["msg 0", "msg 1"], False), (3, ["msg 1", "msg 2"], True)],
)
async def test_session_messages_latest_compacted_exact_boundary_and_one_extra(
    adapter, session_db, logical_count, expected_contents, expected_has_more
):
    session_id = session_db.create_session(
        f"latest-compacted-{logical_count}", "api_server"
    )
    session_db.append_messages_batch(
        session_id,
        [{"role": "user", "content": f"msg {i}"} for i in range(logical_count)],
    )
    _copy_messages_as_compacted_generation(session_db, session_id)
    _copy_messages_as_compacted_generation(session_db, session_id)
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            f"/api/sessions/{session_id}/messages"
            "?include_compacted=true&order=latest&limit=2"
        )
        assert resp.status == 200
        payload = await resp.json()
    assert [message["content"] for message in payload["data"]] == expected_contents
    assert payload["pagination"]["has_more"] is expected_has_more


@pytest.mark.asyncio
async def test_session_messages_rejects_offset_above_sqlite_signed_64_bit(
    adapter, session_db
):
    session_id = session_db.create_session("oversized-offset", "api_server")

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            f"/api/sessions/{session_id}/messages?limit=1&offset=9223372036854775808"
        )
        assert resp.status == 400
        payload = await resp.json()

    assert payload["error"]["code"] == "invalid_pagination"


@pytest.mark.asyncio
async def test_session_messages_accepts_maximum_raw_budget_offset_and_rejects_one_over(
    adapter, session_db
):
    session_id = session_db.create_session("raw-budget-offset", "api_server")
    maximum_safe_offset = SessionDB._DISPLAY_RAW_ROW_BUDGET - 2
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        accepted = await cli.get(
            f"/api/sessions/{session_id}/messages"
            f"?include_compacted=true&limit=1&offset={maximum_safe_offset}"
        )
        rejected = await cli.get(
            f"/api/sessions/{session_id}/messages"
            f"?include_compacted=true&limit=1&offset={maximum_safe_offset + 1}"
        )
        rejected_payload = await rejected.json()
    assert accepted.status == 200
    assert rejected.status == 400
    assert rejected_payload["error"]["code"] == "invalid_pagination"


@pytest.mark.asyncio
async def test_session_messages_compacted_scan_budget_failure_is_bounded_error(
    adapter, session_db, monkeypatch
):
    session_id = session_db.create_session("compacted-work-budget", "api_server")
    session_db.append_messages_batch(
        session_id,
        [{"role": "user", "content": f"msg {i}"} for i in range(12)],
    )
    _copy_messages_as_compacted_generation(session_db, session_id)
    _copy_messages_as_compacted_generation(session_db, session_id)
    monkeypatch.setattr(SessionDB, "_DISPLAY_RAW_ROW_BUDGET", 20)
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(
            f"/api/sessions/{session_id}/messages"
            "?include_compacted=true&order=oldest&limit=4"
        )
        payload = await resp.json()
    assert resp.status == 400
    assert payload["error"]["code"] == "pagination_work_limit"


@pytest.mark.asyncio
async def test_session_messages_zero_limit_does_not_claim_more(adapter, session_db):
    """`limit=0` must not answer "there is more" on a page containing nothing.

    Only negative values are rejected, so `limit=0` is a legal request. Its page is
    necessarily empty, and an empty page trivially has `len(messages) == limit` — so a
    naive full-page test reports `has_more: true` forever. A client that trusts the
    signal to decide whether to keep paging never terminates.
    """
    session_id = session_db.create_session("zero-limit", "api_server")
    session_db.append_messages_batch(
        session_id,
        [{"role": "user", "content": f"msg {i}"} for i in range(3)],
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get(f"/api/sessions/{session_id}/messages?limit=0")
        assert resp.status == 200
        payload = await resp.json()

    assert payload["data"] == []
    assert payload["pagination"]["returned"] == 0
    assert payload["pagination"]["has_more"] is False


@pytest.mark.asyncio
async def test_session_messages_opt_in_exposes_compacted_display_history(
    adapter, session_db
):
    """Rows preserved by in-place compaction are reachable only on request.

    This is precisely why a phone client cannot scroll past the newest
    compaction boundary: the active-only default hides durable transcript the
    user can still legitimately page back to. Opting in must surface it, and
    NOT opting in must leave the existing projection untouched.
    """
    session_id = session_db.create_session("compacted-history", "api_server")
    session_db.append_messages_batch(
        session_id,
        [{"role": "user", "content": f"old {i}"} for i in range(4)],
    )
    session_db.archive_and_compact(
        session_id,
        [
            {"role": "assistant", "content": "summary of old turns"},
            {"role": "user", "content": "live 0"},
        ],
    )

    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        default_resp = await cli.get(f"/api/sessions/{session_id}/messages")
        assert default_resp.status == 200
        default_payload = await default_resp.json()

        opted_resp = await cli.get(
            f"/api/sessions/{session_id}/messages?include_compacted=true"
        )
        assert opted_resp.status == 200
        opted_payload = await opted_resp.json()

    default_contents = [m["content"] for m in default_payload["data"]]
    opted_contents = [m["content"] for m in opted_payload["data"]]

    # Unchanged for every existing caller: the archived generation stays hidden.
    assert not any(c.startswith("old ") for c in default_contents)
    assert default_payload["pagination"]["include_compacted"] is False

    # Opting in reaches back past the compaction boundary...
    assert [c for c in opted_contents if c.startswith("old ")] == [
        "old 0",
        "old 1",
        "old 2",
        "old 3",
    ]
    # ...without losing the live tail, and still chronological.
    assert opted_contents[-len(default_contents):] == default_contents
    assert opted_payload["pagination"]["include_compacted"] is True

    # Every row carries the durable database id the client uses as a stable
    # identity across reloads and page boundaries.
    ids = [m["id"] for m in opted_payload["data"]]
    assert ids and all(isinstance(i, int) for i in ids)
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


@pytest.mark.asyncio
async def test_run_agent_binds_api_session_context_for_tool_env(adapter, monkeypatch):
    """API-server request sessions should reach tools and terminal subprocess env."""
    monkeypatch.setenv("HERMES_SESSION_ID", "stale-session")
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def run_conversation(self, user_message, conversation_history, task_id):
            from gateway.session_context import get_session_env
            from tools.environments.local import _make_run_env

            observed["task_id"] = task_id
            observed["context_session_id"] = get_session_env("HERMES_SESSION_ID")
            observed["context_platform"] = get_session_env("HERMES_SESSION_PLATFORM")
            observed["context_session_key"] = get_session_env("HERMES_SESSION_KEY")
            observed["child_session_id"] = _make_run_env({}).get("HERMES_SESSION_ID")
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        gateway_session_key="request-key",
    )

    assert result["session_id"] == "request-session"
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert "runtime" not in usage
    assert observed == {
        "task_id": "request-session",
        "context_session_id": "request-session",
        "context_platform": "api_server",
        "context_session_key": "request-key",
        "child_session_id": "request-session",
    }


@pytest.mark.asyncio
async def test_run_agent_registers_active_run_id_for_steering(adapter, monkeypatch):
    observed = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, session_id: str):
            self.session_id = session_id

        def steer(self, text: str) -> bool:
            observed["steer_text"] = text
            return True

        def run_conversation(self, user_message, conversation_history, task_id):
            observed["registered"] = adapter._active_run_agents.get("run_steer_test") is self
            observed["task_id"] = task_id
            return {"final_response": "ok"}

    def fake_create_agent(**kwargs):
        return FakeAgent(kwargs["session_id"])

    monkeypatch.setattr(adapter, "_create_agent", fake_create_agent)

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="request-session",
        active_run_id="run_steer_test",
    )

    assert result["session_id"] == "request-session"
    assert usage == {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    assert observed == {"registered": True, "task_id": "request-session"}
    assert "run_steer_test" not in adapter._active_run_agents


@pytest.mark.asyncio
async def test_session_chat_stream_disconnect_keeps_control_refs_until_executor_finishes(
    adapter, session_db
):
    """Disconnects must interrupt the live run without dropping its control refs early."""
    session_id = session_db.create_session("disconnect-stream-session", "api_server")
    run_started = threading.Event()
    interrupt_called = threading.Event()
    allow_finish = threading.Event()
    write_calls = {"count": 0}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, stream_delta_callback):
            self._stream_delta_callback = stream_delta_callback
            self.session_id = session_id

        def interrupt(self, _message=None):
            interrupt_called.set()

        def run_conversation(self, user_message, conversation_history, task_id):
            del user_message, conversation_history, task_id
            run_started.set()
            self._stream_delta_callback("hello")
            allow_finish.wait(timeout=5)
            return {"final_response": "done", "session_id": session_id}

    class DisconnectingStreamResponse:
        async def prepare(self, request):
            del request

        async def write(self, payload):
            del payload
            write_calls["count"] += 1
            if write_calls["count"] >= 3:
                raise ConnectionResetError("simulated client disconnect")

    request = MagicMock()
    request.headers = {}
    request.match_info = {"session_id": session_id}

    def _create_agent(**kwargs):
        return FakeAgent(kwargs["stream_delta_callback"])

    with patch.object(
        adapter,
        "_get_existing_session_or_404",
        return_value=({"id": session_id}, None),
    ), patch.object(
        adapter,
        "_read_json_body",
        return_value=({"message": "stream please"}, None),
    ), patch.object(
        adapter,
        "_create_agent",
        side_effect=_create_agent,
    ), patch(
        "gateway.platforms.api_server.web.StreamResponse",
        return_value=DisconnectingStreamResponse(),
    ):
        handler_task = asyncio.create_task(adapter._handle_session_chat_stream(request))

        for _ in range(60):
            if run_started.is_set():
                break
            await asyncio.sleep(0.05)

        assert run_started.is_set()
        run_id = next(iter(adapter._run_statuses))

        for _ in range(40):
            if interrupt_called.is_set():
                break
            await asyncio.sleep(0.05)

        assert interrupt_called.is_set()
        assert run_id in adapter._active_run_agents
        # Not in _active_run_tasks: session-stream turns are counted via
        # _inflight_agent_runs; a task entry would double-count them in the
        # shutdown drain (active_agent_work_count).
        assert run_id not in adapter._active_run_tasks
        assert not handler_task.done()

        allow_finish.set()
        await handler_task

    assert run_id not in adapter._active_run_agents


@pytest.mark.asyncio
async def test_session_chat_stream_run_completed_carries_turn_transcript(adapter, session_db):
    """run.completed must include the full interleaved turn transcript so a
    client that lost intermediate (pre-tool-call) assistant text from the live
    delta stream can reconcile without a separate /messages fetch. Refs #34703.
    """
    import json as _json

    session_id = session_db.create_session("transcript-session", "api_server")

    async def fake_run(**kwargs):
        # Stream the intermediate planning text the way a real turn would.
        kwargs["stream_delta_callback"]("Let me search for that:")
        kwargs["stream_delta_callback"]("Here is the summary.")
        result = {
            "final_response": "Here is the summary.",
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "search then summarize"},
                {
                    "role": "assistant",
                    "content": "Let me search for that:",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "web_search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "content": "results", "tool_call_id": "call_1", "tool_name": "web_search"},
                {"role": "assistant", "content": "Here is the summary."},
            ],
        }
        return result, {"total_tokens": 6}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "search then summarize"},
            )
            assert resp.status == 200
            body = await resp.text()

    # Pull the run.completed event payload out of the SSE body.
    run_completed_payload = None
    for block in body.split("\n\n"):
        if "event: run.completed" in block:
            for line in block.splitlines():
                if line.startswith("data: "):
                    run_completed_payload = _json.loads(line[len("data: "):])
            break
    assert run_completed_payload is not None, body
    messages = run_completed_payload.get("messages")
    assert isinstance(messages, list) and messages, run_completed_payload

    # The colon-ended intermediate text that preceded the tool call must be present.
    contents = [m.get("content") for m in messages]
    assert "Let me search for that:" in contents
    assert "Here is the summary." in contents
    # No prior-turn user message should leak into the per-turn slice.
    assert all(m.get("role") in ("assistant", "tool") for m in messages)
    # The tool call is preserved alongside the intermediate text.
    assert any(m.get("tool_calls") for m in messages)


# ---------------------------------------------------------------------------
# Session-persisted model threading + provider-auth failure surfacing
# (salvaged from PR #57947 by @FvanW and PR #59941 by @kaishi00)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_session_chat_resolves_stored_model_route_alias(session_db, monkeypatch):
    """A session-persisted model that matches a model_routes alias must go
    through the route path (so route provider/credentials apply) and NOT be
    passed as a raw session_model (idea from PR #59941 by @kaishi00)."""
    adapter = APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={"model_routes": {"alias": {"model": "route/model", "provider": "openrouter"}}},
        )
    )
    adapter._session_db = session_db
    session_id = session_db.create_session("route-pinned-session", "api_server", model="alias")

    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": session_id}, {"total_tokens": 1}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hi"},
            )
            assert resp.status == 200

    _, kwargs = mock_run.call_args
    assert kwargs["route"] == {"model": "route/model", "provider": "openrouter"}
    assert kwargs["session_model"] is None


@pytest.mark.asyncio
async def test_session_chat_treats_pre_existing_poisoned_row_as_no_model(session_db):
    """A session row created before the alias-leak fix may still have the
    virtual model alias (e.g. "hermes-agent") persisted literally as its
    model. Reading that back must NOT thread it through as a raw
    session_model override — it must fall through to the global default,
    exactly like a row that never had a model at all (#session-model-
    alias-leak)."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    session_id = session_db.create_session(
        "poisoned-session", "api_server", model=adapter._model_name
    )

    mock_run = AsyncMock(return_value=({"final_response": "ok", "session_id": session_id}, {"total_tokens": 1}))
    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", mock_run):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "hi"},
            )
            assert resp.status == 200

    _, kwargs = mock_run.call_args
    assert kwargs["session_model"] is None


@pytest.mark.asyncio
async def test_session_chat_stream_treats_pre_existing_poisoned_row_as_no_model(session_db):
    """Streaming twin of the above: the SSE chat path must apply the same
    guard against a pre-existing poisoned session row."""
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    session_id = session_db.create_session(
        "poisoned-stream-session", "api_server", model=adapter._model_name
    )

    async def fake_run(**kwargs):
        return {"final_response": "ok", "session_id": session_id}, {"total_tokens": 1}

    app = _create_session_app(adapter)
    with patch.object(adapter, "_run_agent", side_effect=fake_run) as mock_run:
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "hi"},
            )
            assert resp.status == 200
            # Drain the SSE body: the 200 lands before the streaming task
            # invokes _run_agent, so asserting on call_args without reading
            # the body races the handler (flaked on loaded CI runners).
            await resp.text()

    _, kwargs = mock_run.call_args
    assert kwargs["session_model"] is None


def _register_session_model_route(app, adapter):
    app.router.add_post("/api/sessions/{session_id}/model", adapter._handle_session_model_lock)


def _patch_api_server_runtime(monkeypatch):
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs",
        lambda: {
            "provider": "openrouter",
            "api_key": "sk-global",
            "base_url": "https://openrouter.example/v1",
            "api_mode": "chat_completions",
        },
    )
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "global/model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {})
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_reasoning_config",
        staticmethod(lambda model="": {}),
    )
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: None),
    )
    monkeypatch.setattr("gateway.run._current_max_iterations", lambda: 90)
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: set())
    monkeypatch.setattr(
        "gateway.run._resolve_runtime_agent_kwargs_for_provider",
        lambda provider: {
            "provider": provider,
            "api_key": f"sk-{provider}",
            "base_url": f"https://{provider}.example/v1",
            "api_mode": "chat_completions",
        },
    )


@pytest.mark.asyncio
async def test_create_session_respects_browser_source_and_model_lock(adapter, session_db):
    app = _create_session_app(adapter)
    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post(
            "/api/sessions",
            json={
                "id": "browser-lock-session",
                "source": "hermes_browser",
                "provider": "nous",
                "model": "x-ai/grok-4.5",
                "require_model_lock": True,
                "title": "Browser lock",
                "system_prompt": "browser prompt",
            },
        )
        assert resp.status == 201, await resp.text()
        payload = await resp.json()

    assert payload["session"]["source"] == "hermes_browser"
    assert payload["session"]["model"] == "x-ai/grok-4.5"
    row = session_db.get_session("browser-lock-session")
    assert row["source"] == "hermes_browser"
    assert row["model"] == "x-ai/grok-4.5"
    import json as _json
    model_config = row.get("model_config")
    if isinstance(model_config, str):
        model_config = _json.loads(model_config)
    assert model_config["browser_model_lock"]["provider"] == "nous"
    assert model_config["browser_model_lock"]["model"] == "x-ai/grok-4.5"
    assert model_config["browser_model_lock"]["confirmed"] is True


@pytest.mark.asyncio
async def test_session_model_lock_endpoint_then_chat_reuses_persisted_lock_and_provider_credentials(
    adapter,
    session_db,
    monkeypatch,
):
    session_id = session_db.create_session(
        "endpoint-lock-chat",
        "api_server",
        model="gpt-5.5",
        system_prompt="Conversation started:\nModel: gpt-5.5\nProvider: openai-codex\n",
    )
    captured = {}

    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.session_id = kwargs["session_id"]
            self.provider = kwargs.get("provider") or ""
            self.model = kwargs.get("model") or ""

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "locked", "session_id": self.session_id}

    _patch_api_server_runtime(monkeypatch)
    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr(
        adapter,
        "_session_model_override_for",
        lambda *_: {
            "model": "session/override-model",
            "provider": "openai-codex",
            "api_key": "sk-session-override",
            "base_url": "https://override.example/v1",
            "api_mode": "codex_responses",
        },
    )

    app = _create_session_app(adapter)
    _register_session_model_route(app, adapter)
    with patch.object(adapter, "_resolve_route", return_value=None):
        async with TestClient(TestServer(app)) as cli:
            lock_resp = await cli.post(
                f"/api/sessions/{session_id}/model",
                json={
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "require_model_lock": True,
                },
            )
            assert lock_resp.status == 200, await lock_resp.text()

            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={"message": "use the stored lock"},
            )
            assert resp.status == 200, await resp.text()
            payload = await resp.json()

    assert captured["provider"] == "nous"
    assert captured["model"] == "x-ai/grok-4.5"
    assert captured["api_key"] == "sk-nous"
    assert captured["base_url"] == "https://nous.example/v1"
    assert payload["runtime"]["provider"] == "nous"
    assert payload["runtime"]["model"] == "x-ai/grok-4.5"
    assert payload["runtime"]["requested"] == {
        "provider": "nous",
        "model": "x-ai/grok-4.5",
    }
    assert payload["runtime"]["route_source"] == "session_model_lock"


@pytest.mark.asyncio
async def test_session_model_lock_endpoint_then_chat_stream_reuses_persisted_lock(
    adapter,
    session_db,
):
    session_id = session_db.create_session("endpoint-lock-stream", "api_server")
    captured = {}

    async def fake_run(**kwargs):
        captured.update(kwargs)
        kwargs["stream_delta_callback"]("hi")
        return (
            {
                "final_response": "hi",
                "session_id": session_id,
                "runtime": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "requested": {"provider": "nous", "model": "x-ai/grok-4.5"},
                    "route_source": "session_model_lock",
                },
            },
            {
                "total_tokens": 1,
                "runtime": {
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "requested": {"provider": "nous", "model": "x-ai/grok-4.5"},
                    "route_source": "session_model_lock",
                },
            },
        )

    app = _create_session_app(adapter)
    _register_session_model_route(app, adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(
        adapter,
        "_run_agent",
        side_effect=fake_run,
    ):
        async with TestClient(TestServer(app)) as cli:
            lock_resp = await cli.post(
                f"/api/sessions/{session_id}/model",
                json={
                    "provider": "nous",
                    "model": "x-ai/grok-4.5",
                    "require_model_lock": True,
                },
            )
            assert lock_resp.status == 200, await lock_resp.text()

            resp = await cli.post(
                f"/api/sessions/{session_id}/chat/stream",
                json={"message": "stream with stored lock"},
            )
            assert resp.status == 200, await resp.text()
            body = await resp.text()

    assert captured["route"] == {"provider": "nous", "model": "x-ai/grok-4.5"}
    assert captured["requested_runtime"]["provider"] == "nous"
    assert captured["requested_runtime"]["model"] == "x-ai/grok-4.5"
    assert captured["route_source"] == "session_model_lock"
    assert "x-ai/grok-4.5" in body


@pytest.mark.asyncio
async def test_run_agent_reports_actual_agent_runtime_not_requested_metadata(adapter, monkeypatch):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0

        def __init__(self):
            self.session_id = "runtime-session"
            self.provider = "actual-provider"
            self.model = "actual-model"
            self._hermes_api_runtime = {
                "provider": "requested-provider",
                "model": "requested-model",
                "route_source": "raw_request",
            }

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "ok", "session_id": self.session_id}

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: FakeAgent())

    result, usage = await adapter._run_agent(
        user_message="hello",
        conversation_history=[],
        session_id="runtime-session",
        route={"provider": "requested-provider", "model": "requested-model"},
        requested_runtime={
            "provider": "requested-provider",
            "model": "requested-model",
        },
        route_source="session_model_lock",
    )

    assert result["runtime"]["provider"] == "actual-provider"
    assert result["runtime"]["model"] == "actual-model"
    assert result["runtime"]["requested"] == {
        "provider": "requested-provider",
        "model": "requested-model",
    }
    assert usage["runtime"]["provider"] == "actual-provider"
    assert usage["runtime"]["model"] == "actual-model"


@pytest.mark.asyncio
async def test_confirmed_runtime_lock_rejects_actual_runtime_mismatch(adapter, monkeypatch):
    class FakeAgent:
        session_prompt_tokens = 0
        session_completion_tokens = 0
        session_total_tokens = 0
        session_id = "mismatch-session"
        provider = "fallback-provider"
        model = "fallback-model"

        def run_conversation(self, user_message, conversation_history, task_id):
            return {"final_response": "wrong runtime", "session_id": self.session_id}

    monkeypatch.setattr(adapter, "_create_agent", lambda **kwargs: FakeAgent())

    with pytest.raises(RuntimeError, match="confirmed model lock runtime mismatch"):
        await adapter._run_agent(
            user_message="hello",
            conversation_history=[],
            session_id="mismatch-session",
            route={"provider": "nous", "model": "x-ai/grok-4.5"},
            requested_runtime={"provider": "nous", "model": "x-ai/grok-4.5"},
            route_source="session_model_lock",
            confirmed_runtime_lock=True,
        )


def test_confirmed_runtime_lock_disables_global_fallback_model(adapter, monkeypatch):
    _patch_api_server_runtime(monkeypatch)
    monkeypatch.setattr(
        "gateway.run.GatewayRunner._load_fallback_model",
        staticmethod(lambda: "openrouter/fallback-model"),
    )
    captured = {}

    class FakeAgent:
        provider = "nous"
        model = "x-ai/grok-4.5"

        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)

    adapter._create_agent(
        session_id="locked-session",
        route={"provider": "nous", "model": "x-ai/grok-4.5"},
        confirmed_runtime_lock=True,
    )

    assert captured["fallback_model"] is None


@pytest.mark.asyncio
async def test_unconfirmed_request_does_not_replace_confirmed_session_lock(adapter, session_db):
    session_id = session_db.create_session("one-off-override", "api_server")
    session_db.update_session_runtime_lock(
        session_id,
        provider="nous",
        model="x-ai/grok-4.5",
        route_source="raw_request",
        confirmed=True,
    )
    mock_run = AsyncMock(
        return_value=(
            {
                "final_response": "ok",
                "session_id": session_id,
                "runtime": {"provider": "openrouter", "model": "anthropic/claude-sonnet"},
            },
            {"total_tokens": 1},
        )
    )
    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(
        adapter,
        "_run_agent",
        mock_run,
    ):
        async with TestClient(TestServer(app)) as cli:
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "one turn only",
                    "provider": "openrouter",
                    "model": "anthropic/claude-sonnet",
                },
            )
            assert resp.status == 200, await resp.text()

    import json as _json

    row = session_db.get_session(session_id)
    config = row["model_config"]
    if isinstance(config, str):
        config = _json.loads(config)
    assert config["browser_model_lock"]["provider"] == "nous"
    assert config["browser_model_lock"]["model"] == "x-ai/grok-4.5"
    assert config["browser_model_lock"]["confirmed"] is True


@pytest.mark.asyncio
async def test_require_model_lock_hard_fails_when_global_default_would_be_used(adapter, session_db, monkeypatch):
    session_id = session_db.create_session("lock-fail-session", "api_server")
    monkeypatch.setattr(adapter, "_model_name", "gpt-5.5")
    app = _create_session_app(adapter)
    with patch.object(adapter, "_resolve_route", return_value=None), patch.object(adapter, "_run_agent", new_callable=AsyncMock) as mock_run:
        async with TestClient(TestServer(app)) as cli:
            # empty model + require_model_lock must not silently fall through
            resp = await cli.post(
                f"/api/sessions/{session_id}/chat",
                json={
                    "message": "hello",
                    "provider": "nous",
                    "model": "",
                    "require_model_lock": True,
                },
            )
            assert resp.status in (400, 409), await resp.text()
            body = await resp.json()
            assert body["error"]["code"] in {"model_lock_unavailable", "invalid_model_lock", "missing_model"}
    mock_run.assert_not_called()


@pytest.mark.asyncio
async def test_patch_session_persists_pinned_and_archived(adapter, session_db):
    """PATCH must accept the durable pin/archive flags and round-trip them.

    These were rejected as unsupported fields, so every pin the desktop made
    400'd silently (the client swallows the error) and the pin only ever lived
    in that one app's localStorage. The auto-archive sweep reads
    `sessions.pinned` server-side, so an unpersisted pin does not protect the
    chat it was supposed to keep.
    """
    session_id = session_db.create_session("pin-session", "api_server")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.patch(f"/api/sessions/{session_id}", json={"pinned": True})
        assert resp.status == 200, await resp.text()
        assert (await resp.json())["session"]["pinned"] is True

        # The flag is durable, not just echoed back from the request body.
        assert bool(session_db.get_session(session_id)["pinned"]) is True

        resp = await cli.get(f"/api/sessions/{session_id}")
        assert (await resp.json())["session"]["pinned"] is True

        resp = await cli.patch(f"/api/sessions/{session_id}", json={"pinned": False})
        assert (await resp.json())["session"]["pinned"] is False
        assert bool(session_db.get_session(session_id)["pinned"]) is False

        resp = await cli.patch(f"/api/sessions/{session_id}", json={"archived": True})
        assert (await resp.json())["session"]["archived"] is True
        assert bool(session_db.get_session(session_id)["archived"]) is True


@pytest.mark.asyncio
async def test_patch_session_rejects_non_boolean_pinned(adapter, session_db):
    session_id = session_db.create_session("pin-type-session", "api_server")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.patch(f"/api/sessions/{session_id}", json={"pinned": "yes"})
        assert resp.status == 400, await resp.text()
        assert (await resp.json())["error"]["code"] == "invalid_session_field"


@pytest.mark.asyncio
async def test_patch_session_still_rejects_unknown_fields(adapter, session_db):
    session_id = session_db.create_session("unknown-field-session", "api_server")
    app = _create_session_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.patch(f"/api/sessions/{session_id}", json={"nonsense": 1})
        assert resp.status == 400, await resp.text()
        assert (await resp.json())["error"]["code"] == "unsupported_session_field"
