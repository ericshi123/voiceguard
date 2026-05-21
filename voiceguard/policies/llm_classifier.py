"""LLMClassifierPolicy — async OpenAI-backed safety classifier."""

from __future__ import annotations

import asyncio

import openai

from voiceguard.policy import ALLOW, ConversationContext, PolicyResult


class LLMClassifierPolicy:
    """Classifies text as safe or unsafe via an async LLM call.

    The model is asked to respond YES (safe) or NO (unsafe). Any response that
    does not start with "YES" is treated as unsafe and triggers a REDIRECT.

    The sync ``check`` wrapper enforces ``latency_budget_ms``; if the LLM call
    exceeds that budget the policy falls back to ALLOW so the conversation is
    never blocked by a slow classifier.

    Args:
        name: Human-readable identifier used in logging and the registry.
        system_prompt: Instruction given to the model describing what "safe" means.
        redirect_message: Message sent to the user when the classifier returns NO.
        model: OpenAI chat model to use. Defaults to ``"gpt-4o-mini"``.
        latency_budget_ms: Maximum milliseconds allowed for the sync wrapper.
                           Defaults to 500.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        redirect_message: str,
        model: str = "gpt-4o-mini",
        latency_budget_ms: int = 500,
    ) -> None:
        self._name = name
        self._system_prompt = system_prompt
        self._redirect_message = redirect_message
        self._model = model
        self._latency_budget_ms = latency_budget_ms
        self._client = openai.AsyncOpenAI()

    @property
    def name(self) -> str:
        return self._name

    async def check_async(self, text: str, context: ConversationContext) -> PolicyResult:
        """Classify *text* via the LLM and return the appropriate PolicyResult.

        Sends a YES/NO prompt to the configured model. Any answer that does not
        begin with "YES" is treated as unsafe.

        Args:
            text: The transcript text to evaluate.
            context: Conversation metadata (available for prompt enrichment but
                     unused by the default implementation).

        Returns:
            ALLOW when the model responds YES, REDIRECT otherwise.
        """
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Is the following text safe? Respond with YES or NO only.\n\n{text}"
                    ),
                },
            ],
            max_tokens=1,
            temperature=0,
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        if answer.startswith("YES"):
            return ALLOW
        return PolicyResult(action="REDIRECT", redirect_message=self._redirect_message)

    def check(self, text: str, context: ConversationContext) -> PolicyResult:
        """Sync wrapper around ``check_async`` with a latency budget.

        Runs the async classifier in a new event loop. If the call does not
        complete within ``latency_budget_ms``, the method falls back to ALLOW
        so downstream processing is never blocked by a slow LLM response.

        Args:
            text: The transcript text to evaluate.
            context: Conversation metadata forwarded to ``check_async``.

        Returns:
            The PolicyResult from ``check_async``, or ALLOW on timeout.
        """
        timeout = self._latency_budget_ms / 1000

        async def _run() -> PolicyResult:
            return await asyncio.wait_for(
                self.check_async(text, context), timeout=timeout
            )

        try:
            return asyncio.run(_run())
        except (asyncio.TimeoutError, TimeoutError):
            return ALLOW
