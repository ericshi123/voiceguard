"""GoldSetEvaluator — batch evaluation of policies against labelled examples."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from voiceguard.policy import ConversationContext, PolicyResult
from voiceguard.registry import PolicyRegistry

logger = logging.getLogger(__name__)


@dataclass
class GoldExample:
    """A single labelled test case.

    Attributes:
        text: The transcript to evaluate.
        role: "user" or "assistant".
        expected_action: The ground-truth action ("ALLOW", "BLOCK", "REDIRECT").
        turn_id: Optional turn counter; defaults to 0.
        history: Prior conversation turns.
        metadata: Arbitrary extra fields (e.g. source, category).
    """

    text: str
    role: str
    expected_action: str
    turn_id: int = 0
    history: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Aggregate results from a gold-set run.

    Attributes:
        total: Number of examples evaluated.
        correct: Number where predicted action matched expected.
        false_positives: ALLOW predicted, non-ALLOW expected.
        false_negatives: non-ALLOW predicted, ALLOW expected.
        details: Per-example result rows.
    """

    total: int = 0
    correct: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Fraction of examples where the prediction matched ground truth."""
        return self.correct / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        """TP / (TP + FP) — proportion of non-ALLOW predictions that are correct."""
        tp = self.correct - sum(
            1 for d in self.details if d["expected"] == "ALLOW" and d["predicted"] == "ALLOW"
        )
        denominator = tp + self.false_positives
        return tp / denominator if denominator else 0.0


class GoldSetEvaluator:
    """Evaluate a :class:`~voiceguard.registry.PolicyRegistry` against labelled examples.

    Args:
        registry: The policy registry to test.

    Example::

        evaluator = GoldSetEvaluator(registry)
        examples = GoldSetEvaluator.load_jsonl(Path("gold_set.jsonl"))
        report = await evaluator.run(examples)
        print(f"Accuracy: {report.accuracy:.1%}")
    """

    def __init__(self, registry: PolicyRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def load_jsonl(path: Path) -> list[GoldExample]:
        """Load examples from a JSONL file.

        Each line must be a JSON object with at minimum ``text``, ``role``,
        and ``expected_action`` keys.

        Args:
            path: Path to the JSONL file.

        Returns:
            List of :class:`GoldExample` instances.
        """
        examples: list[GoldExample] = []
        with path.open() as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                try:
                    obj = json.loads(raw)
                    examples.append(
                        GoldExample(
                            text=obj["text"],
                            role=obj["role"],
                            expected_action=obj["expected_action"],
                            turn_id=obj.get("turn_id", 0),
                            history=obj.get("history", []),
                            metadata=obj.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.warning("Skipping malformed line %d: %s", lineno, exc)
        return examples

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def run(
        self,
        examples: list[GoldExample],
        concurrency: int = 16,
    ) -> EvaluationReport:
        """Run the registry against all *examples* and return an aggregate report.

        Args:
            examples: Gold-labelled test cases.
            concurrency: Maximum number of concurrent evaluations. Useful when
                         policies perform async I/O (e.g. LLM judges).

        Returns:
            :class:`EvaluationReport` with accuracy and per-example details.
        """
        report = EvaluationReport(total=len(examples))
        semaphore = asyncio.Semaphore(concurrency)

        async def _evaluate_one(ex: GoldExample) -> dict[str, Any]:
            async with semaphore:
                context = ConversationContext(
                    turn_id=ex.turn_id,
                    role=ex.role,  # type: ignore[arg-type]
                    history=ex.history,
                )
                # PolicyRegistry.evaluate is synchronous; wrap in executor if
                # needed for blocking policies
                result: PolicyResult = await asyncio.get_event_loop().run_in_executor(
                    None, self._registry.evaluate, ex.text, context
                )
                return {
                    "text": ex.text,
                    "expected": ex.expected_action,
                    "predicted": result.action,
                    "match": result.action == ex.expected_action,
                    "metadata": ex.metadata,
                }

        results = await asyncio.gather(*[_evaluate_one(ex) for ex in examples])

        for row in results:
            report.details.append(row)
            if row["match"]:
                report.correct += 1
            elif row["expected"] == "ALLOW":
                # Predicted non-ALLOW when ground truth is ALLOW → false positive
                report.false_positives += 1
            else:
                # Predicted ALLOW when ground truth is non-ALLOW → false negative
                report.false_negatives += 1

        logger.info(
            "GoldSet run complete: accuracy=%.1f%% fp=%d fn=%d total=%d",
            report.accuracy * 100,
            report.false_positives,
            report.false_negatives,
            report.total,
        )
        return report
