"""GuardedService — bidirectional guardrail middleware for OpenAI Realtime API sessions."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, AsyncIterator

from .policy import ConversationContext, PolicyResult
from .registry import PolicyRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class GuardedService:
    """Wraps a ``SessionEventWrapper`` and applies input + output guardrails.

    The service intercepts the async event stream from the Realtime API and:

    * **Input guardrail**: accumulates ``UserTranscriptDelta`` events, checks
      the growing transcript against ``input_policies``, and if a violation is
      found calls ``cancel_response()``, ``delete_item()``, sends a redirect
      message, and sets a flag to suppress a spurious output guardrail trigger.
    * **Output guardrail**: accumulates ``AssistantTranscriptDelta`` events,
      checks against ``output_policies``, and if a violation is found calls
      ``cancel_response()``, ``truncate_assistant()``, and sends a redirect.

    Race-condition safety: when the input guardrail fires it pre-sets
    ``_output_guardrail_suppressed`` so that a partially-generated assistant
    transcript that was already in-flight does not cause a second redirect.

    Args:
        session: An object exposing the ``SessionEventWrapper`` interface (must
                 provide ``get_events()``, ``cancel_response()``,
                 ``delete_item()``, ``truncate_assistant()``, and
                 ``send_text()``).
        input_policies: Registry evaluated against user speech transcripts.
        output_policies: Registry evaluated against assistant transcripts.
    """

    def __init__(
        self,
        session: Any,
        input_policies: PolicyRegistry,
        output_policies: PolicyRegistry,
    ) -> None:
        self._session = session
        self._input_policies = input_policies
        self._output_policies = output_policies

        # Per-turn mutable state — reset on each UserSpeechStarted
        self._user_transcript_buffer: str = ""
        self._input_violation_fired: bool = False
        self._output_guardrail_suppressed: bool = False

        # Output side state
        self._assistant_transcript_buffer: str = ""
        self._current_assistant_item_id: str | None = None
        self._current_assistant_audio_ms: int = 0

        # Turn counter (incremented on UserSpeechStarted)
        self._turn_id: int = 0
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------
    # Main event loop
    # ------------------------------------------------------------------

    async def get_events(self) -> AsyncIterator[Any]:
        """Yield filtered/transformed events from the underlying session.

        Transparently passes through all events that are not suppressed by a
        guardrail violation. Guardrail side-effects (cancel, redirect message)
        are dispatched as async fire-and-forget tasks so as not to block the
        event stream.

        Yields:
            Realtime API event objects (same types as the underlying session).
        """
        async for event in self._session.get_events():
            event_type: str = getattr(event, "type", "")

            if event_type == "UserSpeechStarted":
                self._on_user_speech_started()
                yield event

            elif event_type == "UserTranscriptDelta":
                delta: str = getattr(event, "delta", "")
                self._user_transcript_buffer += delta
                context = self._make_context("user")
                result = self._input_policies.evaluate(self._user_transcript_buffer, context)
                if result.action != "ALLOW" and not self._input_violation_fired:
                    self._input_violation_fired = True
                    # Suppress output guardrail for in-flight assistant audio
                    self._output_guardrail_suppressed = True
                    asyncio.ensure_future(self._handle_input_violation(result, event))
                    # Drop this event — don't yield to caller
                    continue
                if not self._input_violation_fired:
                    yield event

            elif event_type == "UserTranscriptDone":
                if not self._input_violation_fired:
                    # Final check on the complete transcript
                    context = self._make_context("user")
                    result = self._input_policies.evaluate(self._user_transcript_buffer, context)
                    if result.action != "ALLOW":
                        self._input_violation_fired = True
                        self._output_guardrail_suppressed = True
                        asyncio.ensure_future(self._handle_input_violation(result, event))
                        continue
                    # Clean turn — record history and flush
                    self._history.append({"role": "user", "text": self._user_transcript_buffer})
                    yield event
                # else: suppress done event after a violation

            elif event_type == "AssistantTranscriptDelta":
                delta = getattr(event, "delta", "")
                item_id = getattr(event, "item_id", None)
                audio_ms = getattr(event, "audio_end_ms", 0)
                self._assistant_transcript_buffer += delta
                if item_id:
                    self._current_assistant_item_id = item_id
                if audio_ms:
                    self._current_assistant_audio_ms = audio_ms

                if not self._output_guardrail_suppressed:
                    context = self._make_context("assistant")
                    result = self._output_policies.evaluate(
                        self._assistant_transcript_buffer, context
                    )
                    if result.action != "ALLOW":
                        self._output_guardrail_suppressed = True
                        asyncio.ensure_future(self._handle_output_violation(result))
                        continue
                yield event

            elif event_type == "AssistantTranscriptDone":
                assistant_text = getattr(event, "transcript", self._assistant_transcript_buffer)
                self._history.append({"role": "assistant", "text": assistant_text})
                self._assistant_transcript_buffer = ""
                self._output_guardrail_suppressed = False
                yield event

            else:
                yield event

    # ------------------------------------------------------------------
    # Guardrail handlers
    # ------------------------------------------------------------------

    async def _handle_input_violation(self, result: PolicyResult, _event: Any) -> None:
        """Cancel the in-progress response, delete the offending item, and redirect."""
        logger.warning(
            "Input guardrail triggered turn_id=%d action=%s", self._turn_id, result.action
        )
        try:
            await self._session.cancel_response()
        except Exception:
            logger.debug("cancel_response failed (may already be cancelled)", exc_info=True)

        try:
            await self._session.delete_item(self._user_transcript_buffer)
        except Exception:
            logger.debug("delete_item failed", exc_info=True)

        if result.redirect_message:
            await self._session.send_text(result.redirect_message)

    async def _handle_output_violation(self, result: PolicyResult) -> None:
        """Cancel the in-progress response, truncate audio, and send a redirect."""
        logger.warning(
            "Output guardrail triggered turn_id=%d action=%s", self._turn_id, result.action
        )
        try:
            await self._session.cancel_response()
        except Exception:
            logger.debug("cancel_response failed", exc_info=True)

        if self._current_assistant_item_id:
            try:
                await self._session.truncate_assistant(
                    self._current_assistant_item_id,
                    self._current_assistant_audio_ms,
                )
            except Exception:
                logger.debug("truncate_assistant failed", exc_info=True)

        if result.redirect_message:
            await self._session.send_text(result.redirect_message)

        self._assistant_transcript_buffer = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_user_speech_started(self) -> None:
        self._turn_id += 1
        self._user_transcript_buffer = ""
        self._input_violation_fired = False
        self._output_guardrail_suppressed = False
        self._assistant_transcript_buffer = ""
        self._current_assistant_item_id = None
        self._current_assistant_audio_ms = 0

    def _make_context(self, role: str) -> ConversationContext:
        return ConversationContext(
            turn_id=self._turn_id,
            role=role,  # type: ignore[arg-type]
            history=list(self._history),
        )
