"""
Reference example: wrapping an OpenAI Realtime session with VoiceGuard.

This file shows the minimal wiring needed to add bidirectional guardrails
to a voice AI assistant built on the OpenAI Realtime API.
"""

from __future__ import annotations

from typing import Any

# Step 1 — import the three public building blocks from VoiceGuard.
from voiceguard import GuardedService, KeywordPolicy, PolicyRegistry


# ---------------------------------------------------------------------------
# Step 2 — define the input registry (user → model direction).
#
# The input registry is evaluated against every transcript chunk the user
# speaks.  If a policy fires, VoiceGuard cancels the in-progress model
# response, deletes the offending user turn, and sends the redirect message
# back to the caller instead.
# ---------------------------------------------------------------------------

input_registry = PolicyRegistry()

# Block users from mentioning competitor brands by name.
input_registry.add(
    KeywordPolicy(
        name="no-competitor-input",
        patterns=["competitor", "rival brand"],
        redirect_message="I can only help with AcmeCorp products.",
    )
)


# ---------------------------------------------------------------------------
# Step 3 — define the output registry (model → user direction).
#
# The output registry is evaluated against transcript chunks the model
# produces.  If a policy fires, VoiceGuard cancels the response, truncates
# any audio already sent, and sends the redirect message instead.
#
# Reusing the same keyword list here catches cases where the model
# hallucinates a competitor mention despite a system-prompt instruction.
# ---------------------------------------------------------------------------

output_registry = PolicyRegistry()

# TESTING NOTE ---------------------------------------------------------------
# For guardrail testing, do NOT add OpenAI Realtime session instructions that
# tell the model to avoid saying the monitored words (e.g. do NOT set
# instructions="Never mention competitor or rival brand" on your session
# config).  The guardrail — not the model prompt — must be the only line of
# defence; otherwise the model self-censors before the guardrail ever fires,
# making it impossible to verify that the output guardrail actually works.
#
# The patterns below include "openai" and "anthropic" as easy trigger words:
# ask the assistant "what AI companies do you know?" and it will naturally say
# one of them, letting you confirm the guardrail fires without adversarial
# prompt engineering.
# ----------------------------------------------------------------------------

output_registry.add(
    KeywordPolicy(
        name="no-competitor-output",
        patterns=["competitor", "rival brand", "openai", "anthropic"],
        redirect_message="I can only help with AcmeCorp products.",
    )
)


# ---------------------------------------------------------------------------
# Step 4 — wrap a session with GuardedService.
#
# Replace `placeholder_session` with your real SessionEventWrapper object.
# GuardedService.get_events() is a drop-in replacement for the underlying
# session's event iterator — iterate over it exactly as you would before,
# and guardrail enforcement happens transparently.
# ---------------------------------------------------------------------------

def build_guarded_service(session: Any, audio_delay_ms: int = 400) -> GuardedService:
    """Wrap *session* with input and output guardrails.

    Args:
        session: A ``SessionEventWrapper`` from the openai-agents SDK.
        audio_delay_ms: Milliseconds to buffer audio delta events before
            delivering them to the client.  The default (400 ms) gives the
            output guardrail enough lead time to detect a violation in the
            transcript before the corresponding audio reaches the speaker.
            Set to 0 to disable buffering (restores the original behaviour).
    """
    return GuardedService(
        session=session,
        input_policies=input_registry,
        output_policies=output_registry,
        audio_delay_ms=audio_delay_ms,
    )


# ---------------------------------------------------------------------------
# Usage example (shown at import time when run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        "VoiceGuard reference example\n"
        "============================\n"
        "\n"
        "To wire this to the OpenAI Realtime API:\n"
        "\n"
        "  1. Create your RealtimeSession (openai-agents SDK or raw WebSocket wrapper).\n"
        "  2. Call build_guarded_service(session) to get a GuardedService.\n"
        "  3. Replace every `async for event in session.get_events()` loop with\n"
        "     `async for event in guarded.get_events()` — no other changes needed.\n"
        "\n"
        "The registries defined in this file will block any turn (user or assistant)\n"
        "that contains 'competitor', 'rival brand', 'openai', or 'anthropic', and\n"
        "redirect with: 'I can only help with AcmeCorp products.'\n"
        "\n"
        "Testing the output guardrail\n"
        "-----------------------------\n"
        "There is a race condition between when the guardrail detects a forbidden\n"
        "word in the assistant transcript and when the corresponding audio chunks\n"
        "have already been delivered to the client.  GuardedService addresses this\n"
        "with a client-side audio delay buffer (audio_delay_ms, default 400 ms):\n"
        "audio delta events are held for that duration before being yielded, so a\n"
        "transcript-based violation can be caught and the buffered audio discarded\n"
        "before it reaches the speaker.\n"
        "\n"
        "To adjust or disable the buffer:\n"
        "  build_guarded_service(session, audio_delay_ms=0)   # disable\n"
        "  build_guarded_service(session, audio_delay_ms=600) # extra headroom\n"
    )
