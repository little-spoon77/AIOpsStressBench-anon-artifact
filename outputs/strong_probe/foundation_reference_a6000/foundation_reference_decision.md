# Foundation Reference Decision

## Verdict

Use Chronos-Bolt as a bounded reference table or artifact evidence, not as a core winner table model.

## Evidence Summary

### alibaba2018
- `autogluon/chronos-bolt-base`: clean MSE 0.391322, worst stress `delayed_12` MSE 0.689745, P95 latency 21.42 ms, peak memory 404 MB.
- `autogluon/chronos-bolt-small`: clean MSE 0.387705, worst stress `delayed_12` MSE 0.672758, P95 latency 13.28 ms, peak memory 103 MB.
- `autogluon/chronos-bolt-tiny`: clean MSE 0.392137, worst stress `delayed_12` MSE 0.687198, P95 latency 8.74 ms, peak memory 26 MB.

### salesforce_borg
- `autogluon/chronos-bolt-base`: clean MSE 4.71991e-06, worst stress `delayed_12` MSE 1.01098, P95 latency 20.59 ms, peak memory 404 MB.
- `autogluon/chronos-bolt-small`: clean MSE 4.71991e-06, worst stress `delayed_12` MSE 1.06816, P95 latency 12.01 ms, peak memory 103 MB.
- `autogluon/chronos-bolt-tiny`: clean MSE 4.71991e-06, worst stress `delayed_12` MSE 1.14293, P95 latency 8.63 ms, peak memory 26 MB.

## Paper Use

- Supports the claim that foundation models are not a free robustness fix under strict online latency.
- Suitable for a small reference table or appendix/artifact, because the setup is zero-shot and not fine-tuned.
- Do not include Chronos-Bolt in the core comparable winner pool.
