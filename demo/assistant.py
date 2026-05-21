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

output_registry.add(
    KeywordPolicy(
        name="no-competitor-output",
        patterns=["competitor", "rival brand"],
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

def build_guarded_service(session: Any) -> GuardedService:
    """Wrap *session* with input and output guardrails."""
    return GuardedService(
        session=session,
        input_policies=input_registry,
        output_policies=output_registry,
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
        "that contains the words 'competitor' or 'rival brand', and redirect with:\n"
        "  'I can only help with AcmeCorp products.'\n"
    )
