# VoiceGuard

Configurable safety middleware for voice assistants built on the OpenAI Realtime API.

## Overview

VoiceGuard provides bidirectional guardrails for voice assistant sessions — screening both user input and assistant output in real time. It uses a pluggable policy system so teams can define and compose safety rules without modifying core infrastructure. Detection runs on realtime transcripts, keeping added latency minimal for latency-sensitive voice applications.

## Design Document

[VoiceGuard Design Doc](https://docs.google.com/document/d/1pemJG6yUiTKd2mFWsmwXsdrO9nTPUzlFVLATz05uD6k/edit)

## Quick Start

```bash
pip install voiceguard
```

```python
from voiceguard import GuardedService
from voiceguard.policies import KeywordPolicy

service = GuardedService()
service.register(KeywordPolicy(blocked=["credit card number", "social security"]))

async with service.session() as session:
    await session.run()
```

## Architecture

VoiceGuard is built around three core components:

- **PolicyRegistry** — maintains the ordered list of active `SafetyPolicy` instances and dispatches transcript events to each policy in sequence.
- **SafetyPolicy** — base class for all policies; subclasses implement `check_input` and `check_output` to return allow/block/redact decisions.
- **GuardedService** — wraps an OpenAI Realtime API session, intercepts bidirectional audio transcript events, routes them through the `PolicyRegistry`, and enforces decisions before audio is forwarded.

## Evaluation

VoiceGuard evaluations run against a gold set of labeled input/output pairs covering common voice-assistant safety scenarios. HarmBench prompts are adapted for spoken-language context and used as adversarial inputs. Automated judging uses LlamaGuard and WildGuard to score policy decisions at scale, with human review reserved for disagreements between the two judges. Metrics are tracked per policy and per category so regressions surface quickly during development.

## Project Status

Phase 1 in progress.
