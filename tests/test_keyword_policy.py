"""Tests for KeywordPolicy."""
import pytest
from voiceguard.policies.keyword import KeywordPolicy
from voiceguard.policy import ConversationContext

CTX = ConversationContext(turn_id=1, role="user")


def make_policy(patterns=None, redirect="Stay on topic."):
    return KeywordPolicy("test", patterns or [r"forbidden"], redirect)


def test_matching_pattern_redirects():
    p = make_policy([r"forbidden"])
    result = p.check("this is forbidden content", CTX)
    assert result.action == "REDIRECT"
    assert result.redirect_message == "Stay on topic."


def test_no_match_allows():
    p = make_policy([r"forbidden"])
    result = p.check("this is fine", CTX)
    assert result.action == "ALLOW"


def test_case_insensitive():
    p = make_policy([r"forbidden"])
    assert p.check("FORBIDDEN", CTX).action == "REDIRECT"
    assert p.check("Forbidden", CTX).action == "REDIRECT"


def test_multiple_patterns_any_match():
    p = make_policy([r"apple", r"banana"])
    assert p.check("I like banana", CTX).action == "REDIRECT"
    assert p.check("I like apple", CTX).action == "REDIRECT"
    assert p.check("I like mango", CTX).action == "ALLOW"


def test_regex_pattern():
    p = make_policy([r"\bsea\s*world\b"])
    assert p.check("SeaWorld is great", CTX).action == "REDIRECT"
    assert p.check("sea world trip", CTX).action == "REDIRECT"
    assert p.check("the sea is beautiful", CTX).action == "ALLOW"


def test_policy_name():
    p = KeywordPolicy("my-policy", [r"x"], "redirect")
    assert p.name == "my-policy"
