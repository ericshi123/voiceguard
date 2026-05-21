"""GoldSetEvaluator — batch evaluation of policies against labelled examples."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from voiceguard.policy import ConversationContext
from voiceguard.registry import PolicyRegistry

logger = logging.getLogger(__name__)


@dataclass
class GoldSetExample:
    text: str
    role: str
    expected_action: str
    policy_name: str
    notes: str


@dataclass
class EvaluationReport:
    total: int
    passed: int
    failed: int
    precision: float
    recall: float
    failures: list[dict] = field(default_factory=list)


class GoldSetEvaluator:
    @staticmethod
    def load(path: str) -> list[GoldSetExample]:
        """Load examples from a JSON file containing a list of example objects."""
        data = json.loads(Path(path).read_text())
        return [
            GoldSetExample(
                text=item["text"],
                role=item["role"],
                expected_action=item["expected_action"],
                policy_name=item.get("policy_name", ""),
                notes=item.get("notes", ""),
            )
            for item in data
        ]

    def evaluate(
        self,
        examples: list[GoldSetExample],
        registry: PolicyRegistry,
    ) -> EvaluationReport:
        """Run the registry against all examples and return an aggregate report.

        Precision = TP / (TP + FP); recall = TP / (TP + FN), where non-ALLOW
        is treated as the positive class.
        """
        failures: list[dict] = []
        tp = fp = fn = passed = 0

        for ex in examples:
            context = ConversationContext(
                turn_id=0,
                role=ex.role,  # type: ignore[arg-type]
            )
            predicted = registry.evaluate(ex.text, context).action

            if predicted == ex.expected_action:
                passed += 1
                if ex.expected_action != "ALLOW":
                    tp += 1
            else:
                failures.append({
                    "text": ex.text,
                    "role": ex.role,
                    "policy_name": ex.policy_name,
                    "expected": ex.expected_action,
                    "predicted": predicted,
                    "notes": ex.notes,
                })
                if ex.expected_action == "ALLOW":
                    fp += 1
                else:
                    fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return EvaluationReport(
            total=len(examples),
            passed=passed,
            failed=len(failures),
            precision=precision,
            recall=recall,
            failures=failures,
        )

    @staticmethod
    def save_report(report: EvaluationReport, path: str) -> None:
        """Write the report to a JSON file."""
        Path(path).write_text(json.dumps(asdict(report), indent=2))


def run_from_file(
    gold_set_path: str,
    registry: PolicyRegistry,
) -> EvaluationReport:
    """Load a gold set from *gold_set_path* and evaluate it against *registry*."""
    evaluator = GoldSetEvaluator()
    examples = GoldSetEvaluator.load(gold_set_path)
    return evaluator.evaluate(examples, registry)
