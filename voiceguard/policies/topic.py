"""TopicBoundaryPolicy — stub for semantic topic-scope enforcement."""

from __future__ import annotations

from voiceguard.policy import ALLOW, ConversationContext, PolicyResult, SafetyPolicy


class TopicBoundaryPolicy:
    """Placeholder policy that enforces a configurable topic boundary.

    Currently returns ALLOW unconditionally. A full implementation would use
    an embedding model or LLM classifier to determine whether *text* is
    within the allowed topic scope and return REDIRECT otherwise.

    Args:
        name: Human-readable identifier.
        allowed_topics: Descriptive list of topics the assistant is allowed
                        to discuss (used as context for the future classifier).
        redirect_message: Message to send when the topic boundary is crossed.
    """

    def __init__(
        self,
        name: str,
        allowed_topics: list[str],
        redirect_message: str,
    ) -> None:
        self._name = name
        self.allowed_topics = allowed_topics
        self.redirect_message = redirect_message

    @property
    def name(self) -> str:
        return self._name

    def check(self, text: str, context: ConversationContext) -> PolicyResult:
        """Stub: always returns ALLOW until classifier is implemented.

        Args:
            text: The transcript text to evaluate.
            context: Conversation metadata.

        Returns:
            ALLOW (stub behaviour).
        """
        # TODO: call embedding/LLM classifier against self.allowed_topics
        return ALLOW
