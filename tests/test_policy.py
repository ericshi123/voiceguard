"""Tests for core policy types."""
import pytest
from voiceguard.policy import PolicyResult, ConversationContext, ALLOW


def test_allow_result():
    r = PolicyResult(action="ALLOW")
    assert r.action == "ALLOW"
    assert r.redirect_message is None


def test_redirect_requires_message():
    with pytest.raises(ValueError):
        PolicyResult(action="REDIRECT")


def test_redirect_with_message():
    r = PolicyResult(action="REDIRECT", redirect_message="Please stay on topic.")
    assert r.action == "REDIRECT"
    assert r.redirect_message == "Please stay on topic."


def test_allow_constant():
    assert ALLOW.action == "ALLOW"


def test_conversation_context_defaults():
    ctx = ConversationContext(turn_id=1, role="user")
    assert ctx.history == []


def test_conversation_context_with_history():
    ctx = ConversationContext(turn_id=2, role="assistant", history=[{"role": "user", "text": "hi"}])
    assert len(ctx.history) == 1
