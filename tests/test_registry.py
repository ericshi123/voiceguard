"""Tests for PolicyRegistry."""
import pytest
from voiceguard.registry import PolicyRegistry
from voiceguard.policy import PolicyResult, ConversationContext, ALLOW


class AllowPolicy:
    name = "allow-all"
    def check(self, text, context): return ALLOW


class BlockPolicy:
    name = "block-all"
    def check(self, text, context): return PolicyResult(action="BLOCK")


class RedirectPolicy:
    name = "redirect-all"
    def check(self, text, context): return PolicyResult(action="REDIRECT", redirect_message="Redirect!")


CTX = ConversationContext(turn_id=1, role="user")


def test_empty_registry_allows():
    r = PolicyRegistry()
    assert r.evaluate("anything", CTX).action == "ALLOW"


def test_single_allow_policy():
    r = PolicyRegistry()
    r.add(AllowPolicy())
    assert r.evaluate("anything", CTX).action == "ALLOW"


def test_single_block_policy():
    r = PolicyRegistry()
    r.add(BlockPolicy())
    assert r.evaluate("anything", CTX).action == "BLOCK"


def test_short_circuits_on_first_violation():
    r = PolicyRegistry()
    r.add(RedirectPolicy())
    r.add(BlockPolicy())  # should never be reached
    result = r.evaluate("anything", CTX)
    assert result.action == "REDIRECT"


def test_allow_then_block():
    r = PolicyRegistry()
    r.add(AllowPolicy())
    r.add(BlockPolicy())
    assert r.evaluate("anything", CTX).action == "BLOCK"


def test_remove_policy():
    r = PolicyRegistry()
    r.add(BlockPolicy())
    assert r.remove("block-all") is True
    assert r.evaluate("anything", CTX).action == "ALLOW"


def test_remove_nonexistent():
    r = PolicyRegistry()
    assert r.remove("nonexistent") is False


def test_len():
    r = PolicyRegistry()
    r.add(AllowPolicy())
    r.add(BlockPolicy())
    assert len(r) == 2
