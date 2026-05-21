"""TopicBoundaryPolicy — keyword-overlap topic-scope guardrail."""

from __future__ import annotations

import re

from voiceguard.policy import ALLOW, ConversationContext, PolicyResult, SafetyPolicy


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


class TopicBoundaryPolicy:
    """Restricts conversation to an allowlist of permitted topics.

    Scoring approach (v1 heuristic, no ML):
        1. All words across ``allowed_topics`` strings are collected into a set.
        2. The input text is tokenized into lowercase alphabetic words.
        3. The score is the fraction of text words that appear in the topic
           word set: ``len(overlap) / len(text_words)``.
        4. If the score falls below ``threshold`` *and* the text contains more
           than five words, the policy returns REDIRECT; otherwise it returns
           ALLOW.

    Limitations:
        - Pure lexical matching — synonyms, paraphrases, and morphological
          variants (e.g. "invest" vs "investment") are not recognised.
        - Short inputs (<= 5 words) are always allowed to avoid false positives
          on greetings and single-word queries.
        - Common words (articles, prepositions) shared between topics and
          off-topic text inflate the score, so topic lists should be specific.
        - No semantic understanding; a sentence can score high by incidentally
          reusing topic vocabulary while being off-topic in meaning.

    Args:
        name: Human-readable identifier used in logging and the registry.
        allowed_topics: Descriptive phrases or keywords representing permitted
                        topics. All words from every string are pooled.
        redirect_message: Message sent to the user when the topic boundary fires.
        threshold: Minimum overlap fraction required to ALLOW. Defaults to 0.6.
    """

    def __init__(
        self,
        name: str,
        allowed_topics: list[str],
        redirect_message: str,
        threshold: float = 0.6,
    ) -> None:
        self._name = name
        self._topic_words: set[str] = {
            word for topic in allowed_topics for word in _tokenize(topic)
        }
        self._redirect_message = redirect_message
        self._threshold = threshold

    @property
    def name(self) -> str:
        return self._name

    def check(self, text: str, context: ConversationContext) -> PolicyResult:
        """Return REDIRECT when text is off-topic, otherwise ALLOW.

        Short texts (five words or fewer) are always allowed. For longer
        inputs, the fraction of text words present in the allowed-topic
        vocabulary is compared against ``threshold``; texts that fall below it
        are redirected.

        Args:
            text: The transcript text to evaluate.
            context: Conversation metadata (available for future context-aware
                     extensions but unused by this heuristic).

        Returns:
            :class:`~voiceguard.policy.PolicyResult` with action ALLOW or REDIRECT.
        """
        words = _tokenize(text)
        if len(words) <= 5:
            return ALLOW

        overlap = sum(1 for w in words if w in self._topic_words)
        score = overlap / len(words)

        if score < self._threshold:
            return PolicyResult(action="REDIRECT", redirect_message=self._redirect_message)
        return ALLOW
