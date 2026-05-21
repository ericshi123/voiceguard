"""HarmGuardPolicy — multi-category harm detection guardrail.

Covers five standard harm taxonomy categories with tiered severity levels,
adapted from the HarmBench taxonomy (cybercrime, harassment, hate speech,
dangerous instructions, and explicit content).
"""
from __future__ import annotations

import re

from voiceguard.policy import ALLOW, ConversationContext, PolicyResult

# ---------------------------------------------------------------------------
# Pattern library — tiered by severity
# "low"    : catches mild / suggestive violations
# "medium" : catches clear violations (default)
# "high"   : catches only severe / explicit violations
# ---------------------------------------------------------------------------

CATEGORY_PATTERNS: dict[str, dict[str, list[str]]] = {
    "hate_speech": {
        "low": [
            r"\b(those people|you people)\b",
            r"\b(go back to)\b",
        ],
        "medium": [
            r"\b(i hate|despise)\s+(all\s+)?(blacks?|whites?|jews?|muslims?|christians?|hispanics?|asians?|gays?|lesbians?|trans\w*)\b",
            r"\b(inferior|subhuman|vermin)\b",
            r"\b(white\s+supremac|ethnic\s+cleansing|racial\s+purity)\b",
        ],
        "high": [
            r"\b(exterminate|genocide|gas\s+the)\b",
            r"\bn[i\*]gg[ae]r\b",
            r"\bk[i\*]ke\b",
            r"\bsp[i\*]c\b",
            r"\bc[h\*]nk\b",
        ],
    },
    "abuse": {
        "low": [
            r"\b(shut up|you('re| are) (stupid|dumb|idiot))\b",
            r"\b(go to hell|screw you)\b",
        ],
        "medium": [
            r"\b(i('ll| will) (hurt|kill|destroy) you)\b",
            r"\b(worthless|piece of (shit|garbage|trash))\b",
            r"\b(you deserve to (die|suffer|rot))\b",
        ],
        "high": [
            r"\b(i('m| am) going to (kill|murder|rape) you)\b",
            r"\b(i know where you live)\b",
            r"\b(i will find you)\b",
        ],
    },
    "violence": {
        "low": [
            r"\b(beat (him|her|them) up)\b",
            r"\b(punch|kick|slap)\s+(him|her|them|someone)\b",
        ],
        "medium": [
            r"\b(how to (make|build|create) a (bomb|explosive|weapon))\b",
            r"\b(instructions? (for|to) (kill|attack|assault))\b",
            r"\b(mass (shooting|casualt|murder))\b",
        ],
        "high": [
            r"\b(step[- ]by[- ]step (guide|instructions?) (to|for) (kill|murder|attack))\b",
            r"\b(how to (build|assemble) (a\s+)?(pipe bomb|ied|improvised explosive))\b",
            r"\b(acquire (illegal\s+)?weapons? (without|bypassing))\b",
        ],
    },
    "self_harm": {
        "low": [
            r"\b(i want to (hurt|harm) myself)\b",
            r"\b(nobody (cares|loves) me)\b",
        ],
        "medium": [
            r"\b(how (to|do i) (cut|hurt) myself)\b",
            r"\b(methods? (of|for) suicide)\b",
            r"\b(i (want|am going) to (kill|end) myself)\b",
        ],
        "high": [
            r"\b(step[- ]by[- ]step (suicide|self[- ]harm) (guide|instructions?))\b",
            r"\b(lethal (dose|amount) of)\b",
            r"\b(most (effective|painless) way to (die|kill myself|end it))\b",
        ],
    },
    "explicit": {
        "low": [
            r"\b(sexy|hot\s+body|nude[sd]?)\b",
        ],
        "medium": [
            r"\b(explicit\s+sexual|pornograph|sex\s+act|sexual\s+content)\b",
            r"\b(describe\s+(having\s+)?sex\b)",
            r"\b(generate\s+(explicit|sexual|adult)\s+content)\b",
        ],
        "high": [
            r"\b(child\s+(pornograph|sexual\s+abuse|grooming))\b",
            r"\b(underage\s+(sex|sexual|nude|explicit))\b",
            r"\b(minor\s+in\s+(sexual|explicit|nude))\b",
        ],
    },
}

_SEVERITY_LEVELS = ("low", "medium", "high")


def _get_active_patterns(category: str, severity: str) -> list[str]:
    """Return all patterns at or above the given severity level."""
    levels = _SEVERITY_LEVELS[_SEVERITY_LEVELS.index(severity):]
    patterns = []
    for level in levels:
        patterns.extend(CATEGORY_PATTERNS[category].get(level, []))
    return patterns


class HarmGuardPolicy:
    """Multi-category harm detection guardrail using regex pattern matching.

    Covers five harm categories aligned with the HarmBench taxonomy:
    hate_speech, abuse, violence, self_harm, explicit.

    Each category has tiered pattern sets (low / medium / high severity).
    The ``severity`` parameter controls the minimum threshold — "medium"
    (default) catches clear violations while avoiding most false positives.

    Note: This is a rule-based v1 implementation. For higher recall on
    adversarial inputs, pair with ``LLMClassifierPolicy`` which uses
    Claude claude-haiku-3-5 as a semantic judge.

    Args:
        name: Human-readable identifier used in logging and registry.
        redirect_message: Message sent to the user when a violation fires.
        enabled_categories: Subset of categories to activate. None = all five.
        severity: Minimum severity level ("low", "medium", or "high").
    """

    def __init__(
        self,
        name: str,
        redirect_message: str,
        enabled_categories: list[str] | None = None,
        severity: str = "medium",
    ) -> None:
        if severity not in _SEVERITY_LEVELS:
            raise ValueError(f"severity must be one of {_SEVERITY_LEVELS}")

        all_categories = list(CATEGORY_PATTERNS.keys())
        categories = enabled_categories if enabled_categories is not None else all_categories

        for cat in categories:
            if cat not in CATEGORY_PATTERNS:
                raise ValueError(f"Unknown category '{cat}'. Valid: {all_categories}")

        self._name = name
        self._redirect_message = redirect_message
        self._compiled: list[re.Pattern[str]] = [
            re.compile(pattern, re.IGNORECASE)
            for cat in categories
            for pattern in _get_active_patterns(cat, severity)
        ]

    @property
    def name(self) -> str:
        return self._name

    def check(self, text: str, context: ConversationContext) -> PolicyResult:
        """Return REDIRECT if text contains a harm pattern, otherwise ALLOW.

        Args:
            text: The transcript text to evaluate.
            context: Conversation metadata (available for future context-aware
                     extensions but unused by this rule-based implementation).

        Returns:
            PolicyResult with action ALLOW or REDIRECT.
        """
        for pattern in self._compiled:
            if pattern.search(text):
                return PolicyResult(
                    action="REDIRECT",
                    redirect_message=self._redirect_message,
                )
        return ALLOW
