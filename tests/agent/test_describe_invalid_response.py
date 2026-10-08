"""``describe_invalid_response`` must not raise on response objects without ``__dict__``.

It runs inside the invalid-response retry path. A provider client that hands back a str, dict,
coroutine or generator used to make ``vars(response)`` raise ``TypeError`` there, and the turn
then failed as a non-retryable error on its first attempt instead of retrying.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from agent.turn_recovery import describe_invalid_response


async def _coro():
    return None


def _gen():
    yield 1


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize("factory", [
    lambda: "plain text body",
    lambda: {"unexpected": "dict"},
    lambda: [1, 2, 3],
    _gen,
], ids=["str", "dict", "list", "generator"])
def test_objects_without_dict_do_not_raise(factory, verbose):
    agent = SimpleNamespace(verbose_logging=verbose)
    error_msg, provider_name, hint = describe_invalid_response(agent, factory(), 1.0)
    assert (error_msg, provider_name) == ("Unknown", "Unknown")
    assert isinstance(hint, str) and hint


def test_coroutine_response_does_not_raise():
    agent = SimpleNamespace(verbose_logging=True)
    coro = _coro()
    try:
        assert describe_invalid_response(agent, coro, 1.0)[:2] == ("Unknown", "Unknown")
    finally:
        coro.close()


def test_verbose_logging_still_reports_public_attributes(caplog):
    agent = SimpleNamespace(verbose_logging=True)
    response = SimpleNamespace(choices=None, id="resp-1", _private="hidden")
    with caplog.at_level(logging.DEBUG):
        describe_invalid_response(agent, response, 1.0)
    text = caplog.text
    assert "resp-1" in text and "hidden" not in text
