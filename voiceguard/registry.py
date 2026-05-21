"""PolicyRegistry — ordered chain of SafetyPolicy evaluators."""

from __future__ import annotations

import logging

from .policy import ALLOW, ConversationContext, PolicyResult, SafetyPolicy

logger = logging.getLogger(__name__)


class PolicyRegistry:
    """Ordered collection of :class:`~voiceguard.policy.SafetyPolicy` objects.

    Policies are evaluated in insertion order. Evaluation short-circuits on
    the first non-ALLOW result so that policies can be prioritised.

    Example::

        registry = PolicyRegistry()
        registry.add(KeywordPolicy("profanity", [...], "Let's keep things friendly."))
        result = registry.evaluate("some text", context)
    """

    def __init__(self) -> None:
        self._policies: list[SafetyPolicy] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, policy: SafetyPolicy) -> None:
        """Append *policy* to the end of the evaluation chain.

        Args:
            policy: Any object satisfying the :class:`~voiceguard.policy.SafetyPolicy`
                    protocol.
        """
        self._policies.append(policy)
        logger.debug("Registered policy '%s' (total=%d)", policy.name, len(self._policies))

    def remove(self, name: str) -> bool:
        """Remove the first policy with the given *name*.

        Args:
            name: The policy's ``name`` attribute.

        Returns:
            ``True`` if a policy was removed, ``False`` if none matched.
        """
        for i, p in enumerate(self._policies):
            if p.name == name:
                self._policies.pop(i)
                logger.debug("Removed policy '%s'", name)
                return True
        return False

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, text: str, context: ConversationContext) -> PolicyResult:
        """Run *text* through every policy and return the first non-ALLOW result.

        If all policies return ALLOW (or the registry is empty) an ALLOW result
        is returned.

        Args:
            text: The text to evaluate.
            context: Current conversation metadata.

        Returns:
            The first non-ALLOW :class:`~voiceguard.policy.PolicyResult`, or
            :data:`~voiceguard.policy.ALLOW` if every policy passes.
        """
        for policy in self._policies:
            result = policy.check(text, context)
            if result.action != "ALLOW":
                logger.info(
                    "Policy '%s' triggered action=%s turn_id=%d role=%s",
                    policy.name,
                    result.action,
                    context.turn_id,
                    context.role,
                )
                return result
        return ALLOW

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    @property
    def policies(self) -> list[SafetyPolicy]:
        """Read-only snapshot of the current policy list (in evaluation order)."""
        return list(self._policies)

    def __len__(self) -> int:
        return len(self._policies)

    def __repr__(self) -> str:
        names = [p.name for p in self._policies]
        return f"PolicyRegistry(policies={names!r})"
