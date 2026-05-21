"""AutomatedJudge — protocol for LLM-based safety classifiers.

Planned integrations: LlamaGuard (Meta) and WildGuard (Allen AI). Both models
accept a text prompt and return a binary safety verdict. Concrete implementations
will be added once the model-serving layer is in place; downstream code should
depend on this protocol so the backend can be swapped without callers changing.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable


@runtime_checkable
class AutomatedJudge(Protocol):
    """Protocol for LLM-based safety classifiers (e.g. LlamaGuard, WildGuard).

    Implementations are duck-typed — no base class required. The ``judge``
    method must be synchronous; async wrappers can be layered on top if the
    underlying model endpoint is async.
    """

    def judge(self, text: str) -> Literal["SAFE", "UNSAFE"]:
        """Classify *text* as safe or unsafe.

        Args:
            text: The transcript fragment or full turn text to evaluate.

        Returns:
            ``"SAFE"`` if the text passes the classifier, ``"UNSAFE"`` otherwise.
        """
        ...
