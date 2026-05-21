# VoiceGuard Demo

## What the demo shows

`demo/assistant.py` is the minimal reference wiring for adding VoiceGuard guardrails to an OpenAI Realtime API voice session. It demonstrates:

- **Input guardrails** — evaluated against every transcript chunk the *user* speaks. If a policy fires, VoiceGuard cancels the in-progress model response and returns a redirect message instead.
- **Output guardrails** — evaluated against transcript chunks the *model* produces. If a policy fires, VoiceGuard cancels the response and truncates any audio already sent.

The demo configures two `KeywordPolicy` instances (one per direction) that block any turn containing `"competitor"` or `"rival brand"` and redirect the caller with:

> "I can only help with AcmeCorp products."

## How to run

```bash
# From the repo root
pip install -e .
python demo/assistant.py
```

Running the script prints the integration guide. To wire it to a real session, pass your `RealtimeSession` object to `build_guarded_service()` and iterate over `guarded.get_events()` instead of the raw session events.

## How the gold-set evaluator works

`GoldSetEvaluator` (in `voiceguard/evaluation/gold_set.py`) runs a `PolicyRegistry` over a list of labeled examples and produces precision/recall metrics.

Each example in the gold set has this shape:

| Field | Type | Description |
|---|---|---|
| `text` | string | Transcript turn to evaluate |
| `role` | `"user"` or `"assistant"` | Direction of the turn |
| `expected_action` | `"ALLOW"`, `"BLOCK"`, or `"REDIRECT"` | Ground-truth label |
| `policy_name` | string | Policy expected to fire (empty if ALLOW) |
| `notes` | string | Human-readable rationale |

Precision and recall treat any non-`ALLOW` outcome as the positive class. A failure entry is written for every example where the predicted action differs from the expected action.

### Running evaluation

```python
from voiceguard import KeywordPolicy, PolicyRegistry
from voiceguard.evaluation import GoldSetEvaluator

# Build the registry you want to test
registry = PolicyRegistry()
registry.add(
    KeywordPolicy(
        name="no-competitor-input",
        patterns=["competitor", "rival brand"],
        redirect_message="I can only help with AcmeCorp products.",
    )
)

# Load examples and evaluate
examples = GoldSetEvaluator.load("evaluation/sample_gold_set.json")
report = GoldSetEvaluator().evaluate(examples, registry)

print(f"Passed:    {report.passed}/{report.total}")
print(f"Precision: {report.precision:.2%}")
print(f"Recall:    {report.recall:.2%}")

if report.failures:
    print("\nFailures:")
    for f in report.failures:
        print(f"  [{f['expected']} → {f['predicted']}] {f['text']!r}")
```

To persist the report:

```python
GoldSetEvaluator.save_report(report, "evaluation/report.json")
```

The sample gold set lives at `evaluation/sample_gold_set.json` and covers 10 labeled examples: competitor mentions (user and assistant directions), off-topic requests, and clean in-scope turns.
