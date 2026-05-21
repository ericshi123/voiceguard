"""Core types for the VoiceGuard safety framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable


@dataclass
class ConversationContext:
    """Snapshot of conversation state at the point of a guardrail check.

    Attributes:
        turn_id: Monotonically increasing integer identifying the current turn.
        role: Whether the current text is from the "user" or "assistant".
        history: List of prior turns, each a dict with "role" and "text" keys.
    """

    turn_id: int
    role: Literal["user", "assistant"]
    history: list[dict[str, str]] = field(default_factory=list)


@dataclass
class PolicyResult:
    """Outcome of a single policy check.

    Attributes:
        action: ALLOW means the text is clean; BLOCK means stop and say nothing;
                REDIRECT means stop and send redirect_message to the user instead.
        redirect_message: Required when action is REDIRECT; ignored otherwise.
    """

    action: Literal["ALLOW", "BLOCK", "REDIRECT"]
    redirect_message: str | None = None

    def __post_init__(self) -> None:
        if self.action == "REDIRECT" and not self.redirect_message:
            raise ValueError("REDIRECT result must include a redirect_message")


ALLOW = PolicyResult(action="ALLOW")


@runtime_checkable
class SafetyPolicy(Protocol):
    """Protocol that every guardrail policy must implement.

    Implementations are duck-typed — no base class required, only the
    ``name`` attribute and ``check`` method.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier for this policy (used in logging/metrics)."""
        ...

    def check(self, text: str, context: ConversationContext) -> PolicyResult:
        """Evaluate *text* in the given conversation context.

        Args:
            text: The transcript fragment or full turn text to evaluate.
            context: Conversation metadata (turn_id, role, prior history).

        Returns:
            A :class:`PolicyResult` indicating the action to take.
        """
        ...
