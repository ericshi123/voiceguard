"""Tests for GuardedService bidirectional guardrails."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from voiceguard.service import GuardedService
from voiceguard.registry import PolicyRegistry
from voiceguard.policy import PolicyResult, ALLOW
from voiceguard.policies.keyword import KeywordPolicy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_event(type_: str, **kwargs):
    ev = MagicMock()
    ev.type = type_
    for k, v in kwargs.items():
        setattr(ev, k, v)
    return ev


async def events_gen(*events):
    for ev in events:
        yield ev


class FakeSession:
    def __init__(self, events):
        self._events = events
        self.cancel_response = AsyncMock()
        self.delete_item = AsyncMock()
        self.truncate_assistant = AsyncMock()
        self.send_text = AsyncMock()

    async def get_events(self):
        for ev in self._events:
            yield ev


def clean_registry() -> PolicyRegistry:
    return PolicyRegistry()


def redirect_registry(pattern: str = r"bad word") -> PolicyRegistry:
    r = PolicyRegistry()
    r.add(KeywordPolicy("test-policy", [pattern], "Please stay on topic."))
    return r


# ---------------------------------------------------------------------------
# Input guardrail tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_input_guardrail_fires_on_violation():
    """UserTranscriptDelta with forbidden text → cancel + redirect sent."""
    events = [
        make_event("UserSpeechStarted"),
        make_event("UserTranscriptDelta", delta="please say bad word now"),
    ]
    session = FakeSession(events)
    svc = GuardedService(session, redirect_registry(), clean_registry())

    collected = []
    async for ev in svc.get_events():
        collected.append(ev)

    # Allow time for fire-and-forget tasks
    await asyncio.sleep(0.01)

    session.cancel_response.assert_called_once()
    session.send_text.assert_called_once_with("Please stay on topic.")
    # The violating delta event should NOT be yielded
    types = [e.type for e in collected]
    assert "UserTranscriptDelta" not in types


@pytest.mark.asyncio
async def test_input_guardrail_clean_turn_passes():
    """Clean user transcript passes through with no side effects."""
    events = [
        make_event("UserSpeechStarted"),
        make_event("UserTranscriptDelta", delta="hello how are you"),
        make_event("UserTranscriptDone", transcript="hello how are you"),
    ]
    session = FakeSession(events)
    svc = GuardedService(session, redirect_registry(), clean_registry())

    collected = []
    async for ev in svc.get_events():
        collected.append(ev)

    await asyncio.sleep(0.01)

    session.cancel_response.assert_not_called()
    session.send_text.assert_not_called()
    types = [e.type for e in collected]
    assert "UserTranscriptDelta" in types
    assert "UserTranscriptDone" in types


# ---------------------------------------------------------------------------
# Output guardrail tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_guardrail_fires_on_violation():
    """AssistantTranscriptDelta with forbidden text → cancel + truncate + redirect."""
    events = [
        make_event("UserSpeechStarted"),
        make_event(
            "AssistantTranscriptDelta",
            delta="the bad word is here",
            item_id="item-1",
            audio_end_ms=500,
        ),
    ]
    session = FakeSession(events)
    svc = GuardedService(session, clean_registry(), redirect_registry())

    collected = []
    async for ev in svc.get_events():
        collected.append(ev)

    await asyncio.sleep(0.01)

    session.cancel_response.assert_called_once()
    session.truncate_assistant.assert_called_once_with("item-1", 500)
    session.send_text.assert_called_once_with("Please stay on topic.")
    # Violating delta should NOT be yielded
    types = [e.type for e in collected]
    assert "AssistantTranscriptDelta" not in types


@pytest.mark.asyncio
async def test_output_guardrail_clean_response_passes():
    """Clean assistant transcript passes through with no side effects."""
    events = [
        make_event("UserSpeechStarted"),
        make_event("AssistantTranscriptDelta", delta="Here is helpful info.", item_id="item-1", audio_end_ms=200),
        make_event("AssistantTranscriptDone", transcript="Here is helpful info."),
    ]
    session = FakeSession(events)
    svc = GuardedService(session, clean_registry(), redirect_registry())

    collected = []
    async for ev in svc.get_events():
        collected.append(ev)

    await asyncio.sleep(0.01)

    session.cancel_response.assert_not_called()
    session.send_text.assert_not_called()
    types = [e.type for e in collected]
    assert "AssistantTranscriptDelta" in types


# ---------------------------------------------------------------------------
# Race condition test
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_race_condition_input_suppresses_output_guardrail():
    """Input guardrail pre-suppresses output so no duplicate redirect is sent."""
    events = [
        make_event("UserSpeechStarted"),
        # Input violation
        make_event("UserTranscriptDelta", delta="bad word in user speech"),
        # In-flight assistant response that also contains violation
        make_event("AssistantTranscriptDelta", delta="bad word in assistant response", item_id="item-1", audio_end_ms=100),
    ]
    session = FakeSession(events)
    svc = GuardedService(session, redirect_registry(), redirect_registry())

    async for _ in svc.get_events():
        pass

    await asyncio.sleep(0.01)

    # send_text should only be called ONCE (from input guardrail, not output)
    assert session.send_text.call_count == 1
    # truncate_assistant should NOT be called since output was suppressed
    session.truncate_assistant.assert_not_called()
