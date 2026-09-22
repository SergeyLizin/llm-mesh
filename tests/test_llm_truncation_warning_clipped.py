"""Truncation warnings must identify the token budget actually sent after clipping, while retaining
the requested value for context. Client-level wire checks live in test_client_transport_contracts.py.
"""

from __future__ import annotations

import logging


from llm_mesh._common import warn_if_truncated
from llm_mesh.types import LLMRequest

_LENGTH_CHOICE = {"finish_reason": "length"}


def _request(max_tokens: int) -> LLMRequest:
    return LLMRequest(system="s", user="u", max_tokens=max_tokens)


def _warn(caplog, **kwargs) -> str:
    caplog.clear()
    logger = logging.getLogger("test.truncation")
    with caplog.at_level(logging.WARNING, logger="test.truncation"):
        warn_if_truncated(
            _LENGTH_CHOICE,
            _request(kwargs.pop("requested")),
            kwargs.pop("content", ""),
            provider="GigaChat",
            logger=logger,
            **kwargs,
        )
    return "\n".join(r.getMessage() for r in caplog.records)


def test_clipped_budget_names_wire_value_not_request(caplog):
    """Report 4096 on the wire when the caller requested 8192."""
    msg = _warn(caplog, requested=8192, sent_max_tokens=4096)
    assert "max_tokens=4096" in msg
    # The old warning incorrectly reported the pre-clip value.
    assert "max_tokens=8192" not in msg
    # Retain the requested value to explain the caller's original budget.
    assert "requested 8192" in msg
    assert "clipped to model limit" in msg


def test_unclipped_warning_reports_budget_and_content_length(caplog):
    msg = _warn(caplog, requested=4096, sent_max_tokens=4096, content="abc")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    assert "max_tokens=4096" in msg
    assert "finish_reason=length" in msg
    assert "completion_len=3" in msg
    assert "clipped" not in msg


def test_without_argument_falls_back_to_request(caplog):
    """Without sent_max_tokens, retain backward-compatible request-budget reporting."""
    msg = _warn(caplog, requested=8192)
    assert "max_tokens=8192" in msg
    assert "clipped to model limit" not in msg


def test_silent_when_finish_reason_is_not_length(caplog):
    """Do not warn for finish reasons other than length."""
    caplog.clear()
    logger = logging.getLogger("test.truncation")
    with caplog.at_level(logging.WARNING, logger="test.truncation"):
        warn_if_truncated(
            {"finish_reason": "stop"},
            _request(8192),
            "",
            provider="GigaChat",
            logger=logger,
            sent_max_tokens=4096,
        )
    assert caplog.records == []


