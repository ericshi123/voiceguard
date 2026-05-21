"""KeywordPolicy — regex-based redirect guardrail."""

from __future__ import annotations

import re

from voiceguard.policy import ConversationContext, PolicyResult, SafetyPolicy


class KeywordPolicy:
    """Blocks (redirects) any text that matches one or more regex patterns.

    The check is case-insensitive and searches anywhere in the text (not just
    at word boundaries), so patterns should be anchored explicitly if needed.

    Args:
        name: Human-readable identifier used in logging and the registry.
        patterns: List of regex strings. Any match triggers a REDIRECT.
        redirect_message: The message sent to the user when a pattern fires.

    Example::

        policy = KeywordPolicy(
            name="no-financial-advice",
            patterns=[r"\\bstock tip\\b", r"\\binvest in\\b"],
            redirect_message="I can't provide financial advice. Try a licensed advisor.",
        )
    """

    def __init__(
        self,
        name: str,
        patterns: list[str],
        redirect_message: str,
    ) -> None:
        self._name = name
        self._compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
        self._redirect_message = redirect_message

    # SafetyPolicy protocol implementation

    @property
    def name(self) -> str:
        return self._name

    def check(self, text: str, context: ConversationContext) -> PolicyResult:
        """Return REDIRECT if any pattern matches *text*, otherwise ALLOW.

        Args:
            text: The transcript text to scan.
            context: Conversation metadata (unused by this policy but available
                     for subclasses that extend context-aware logic).

        Returns:
            :class:`~voiceguard.policy.PolicyResult` with action ALLOW or REDIRECT.
        """
        for pattern in self._compiled:
            if pattern.search(text):
                return PolicyResult(
                    action="REDIRECT",
                    redirect_message=self._redirect_message,
                )
        return PolicyResult(action="ALLOW")
