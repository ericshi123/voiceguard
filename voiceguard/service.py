"""GuardedService — bidirectional guardrail middleware for OpenAI Realtime API sessions."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator

from .policy import ConversationContext, PolicyResult
from .registry import PolicyRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_AUDIO_DELTA_TYPES = frozenset(["response.audio.delta", "AudioDelta", "audio_delta"])


def _is_audio_delta(event_type: str) -> bool:
    if event_type in _AUDIO_DELTA_TYPES:
        return True
    lower = event_type.lower()
    return "audio" in lower and "delta" in lower


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

    Audio delay buffer: audio delta events (``response.audio.delta`` and
    similar) are held in an internal queue for ``audio_delay_ms`` milliseconds
    before being yielded to the caller.  This ensures that when the output
    guardrail detects a harmful word in the transcript, the corresponding audio
    chunks have not yet been delivered to the client.  If a violation fires
    while audio is buffered, ``_flush_audio_buffer()`` discards the entire
    queue so no harmful audio reaches the client.  Set ``audio_delay_ms=0`` to
    disable buffering (audio passes through immediately, restoring the original
    race-condition behaviour).

    Args:
        session: An object exposing the ``SessionEventWrapper`` interface (must
                 provide ``get_events()``, ``cancel_response()``,
                 ``delete_item()``, ``truncate_assistant()``, and
                 ``send_text()``).
        input_policies: Registry evaluated against user speech transcripts.
        output_policies: Registry evaluated against assistant transcripts.
        audio_delay_ms: Milliseconds to hold audio delta events before
                        yielding them.  Defaults to 400.  Set to 0 to disable.
    """

    def __init__(
        self,
        session: Any,
        input_policies: PolicyRegistry,
        output_policies: PolicyRegistry,
        audio_delay_ms: int = 400,
    ) -> None:
        self._session = session
        self._input_policies = input_policies
        self._output_policies = output_policies
        self._audio_delay_ms = audio_delay_ms

        # Per-turn mutable state — reset on each UserSpeechStarted
        self._user_transcript_buffer: str = ""
        self._input_violation_fired: bool = False
        self._output_guardrail_suppressed: bool = False

        # Output side state
        self._assistant_transcript_buffer: str = ""
        self._current_assistant_item_id: str | None = None
        self._current_assistant_audio_ms: int = 0

        # Audio delay buffer: list of (enqueue_time_seconds, event)
        self._audio_buffer: list[tuple[float, Any]] = []
        # Queue used to wake the drain loop when new audio arrives
        self._audio_queue: asyncio.Queue[tuple[float, Any] | None] = asyncio.Queue()

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

        Audio delta events are held in an internal delay buffer for
        ``audio_delay_ms`` milliseconds.  A background drain task yields them
        after the delay, or discards them if a violation has fired.

        Yields:
            Realtime API event objects (same types as the underlying session).
        """
        drain_task: asyncio.Task[None] | None = None
        drain_queue: asyncio.Queue[Any] = asyncio.Queue()

        if self._audio_delay_ms > 0:
            drain_task = asyncio.ensure_future(
                self._audio_drain_loop(drain_queue)
            )

        try:
            # Interleave drained audio events with the main stream.
            # Because the drain loop puts events into drain_queue, we need a
            # merged iteration strategy.  We use a shared output queue that
            # both the main loop and the drain loop push into.
            output_queue: asyncio.Queue[Any | None] = asyncio.Queue()

            # Re-create drain task to push into output_queue instead
            if drain_task is not None:
                drain_task.cancel()
                try:
                    await drain_task
                except (asyncio.CancelledError, Exception):
                    pass

            if self._audio_delay_ms > 0:
                drain_task = asyncio.ensure_future(
                    self._audio_drain_loop(output_queue)
                )

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
                        continue
                    if not self._input_violation_fired:
                        yield event

                elif event_type == "UserTranscriptDone":
                    if not self._input_violation_fired:
                        context = self._make_context("user")
                        result = self._input_policies.evaluate(self._user_transcript_buffer, context)
                        if result.action != "ALLOW":
                            self._input_violation_fired = True
                            self._output_guardrail_suppressed = True
                            asyncio.ensure_future(self._handle_input_violation(result, event))
                            continue
                        self._history.append({"role": "user", "text": self._user_transcript_buffer})
                        yield event

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
                            self._flush_audio_buffer()
                            asyncio.ensure_future(self._handle_output_violation(result))
                            continue
                    yield event

                elif event_type == "AssistantTranscriptDone":
                    assistant_text = getattr(event, "transcript", self._assistant_transcript_buffer)
                    self._history.append({"role": "assistant", "text": assistant_text})
                    self._assistant_transcript_buffer = ""
                    self._output_guardrail_suppressed = False
                    yield event

                elif _is_audio_delta(event_type) and self._audio_delay_ms > 0:
                    # Buffer audio delta — drain loop will yield after the delay
                    self._audio_queue.put_nowait((time.monotonic(), event))

                    # Drain any audio events that have aged out, yielding them
                    while not output_queue.empty():
                        yield output_queue.get_nowait()

                else:
                    yield event

            # Main stream exhausted — drain remaining buffered audio
            if self._audio_delay_ms > 0:
                # Signal drain loop to finish
                self._audio_queue.put_nowait(None)
                if drain_task is not None:
                    try:
                        await asyncio.wait_for(drain_task, timeout=2.0)
                    except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                        pass
                while not output_queue.empty():
                    item = output_queue.get_nowait()
                    if item is not None:
                        yield item

        finally:
            if drain_task is not None and not drain_task.done():
                drain_task.cancel()
                try:
                    await drain_task
                except (asyncio.CancelledError, Exception):
                    pass

    # ------------------------------------------------------------------
    # Guardrail handlers
    # ------------------------------------------------------------------

    async def _audio_drain_loop(self, output_queue: asyncio.Queue[Any]) -> None:
        """Background task: yield buffered audio events after the configured delay.

        Reads from ``_audio_queue``.  Each entry is either a
        ``(timestamp, event)`` tuple or ``None`` (sentinel to stop).  Events
        whose age exceeds ``audio_delay_ms`` are forwarded to *output_queue*
        unless ``_output_guardrail_suppressed`` is set, in which case they are
        discarded.
        """
        delay_s = self._audio_delay_ms / 1000.0
        try:
            while True:
                entry = await self._audio_queue.get()
                if entry is None:
                    # Drain remaining items that have waited long enough
                    while not self._audio_queue.empty():
                        leftover = self._audio_queue.get_nowait()
                        if leftover is None:
                            break
                        ts, ev = leftover
                        if not self._output_guardrail_suppressed:
                            output_queue.put_nowait(ev)
                    break

                ts, ev = entry
                now = time.monotonic()
                wait = delay_s - (now - ts)
                if wait > 0:
                    await asyncio.sleep(wait)

                if not self._output_guardrail_suppressed:
                    output_queue.put_nowait(ev)
        except asyncio.CancelledError:
            pass

    def _flush_audio_buffer(self) -> None:
        """Discard all buffered audio delta events (called on violation)."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

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
        self._flush_audio_buffer()

    def _make_context(self, role: str) -> ConversationContext:
        return ConversationContext(
            turn_id=self._turn_id,
            role=role,  # type: ignore[arg-type]
            history=list(self._history),
        )
