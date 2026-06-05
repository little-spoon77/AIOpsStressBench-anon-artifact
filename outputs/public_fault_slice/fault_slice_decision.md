# Public Fault-Injection Slice Decision

- Scope: public fault-injection slice evidence, not production incident validation.
- Source: RCAEval RE1-OB public fault-injection metrics.
- Models: DLinear, RACE-DLinear, PatchTST-style.

## Decision

Artifact-only / caution.

The probe successfully parses and evaluates RCAEval RE1-OB windows, but it should not be promoted to a main-text claim. Across the evaluated slice, normal and fault windows produce nearly identical forecasting metrics for most model-case rows, so the result does not provide strong evidence that public fault-injection windows reproduce the controlled-stress ranking shifts. Keep this as an artifact defense showing that the probe exists, not as production incident validation or a central result.

## Summary

- Candidate files: 30
- Parsed cases: 30
- Metric rows: 180
- Normal-to-fault MSE ranking changes: 0
- Fault-window MSE-vs-capacity disagreements: 12
- Nonzero normal-vs-fault MSE deltas: 3/90 model-case rows
- Nonzero normal-vs-fault capacity deltas: 3/90 model-case rows

## Paper Use

Do not add this result to the main claim. If mentioned, use only this wording: "An artifact-level RCAEval RE1-OB public fault-injection probe is implemented, but the current slice does not provide strong enough normal-to-fault forecasting separation to serve as main-text validation."

## Skipped Files

- Skipped candidates: 0
