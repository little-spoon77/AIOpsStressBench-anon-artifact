| model | implementation | protocol | coverage | budget | role |
| --- | --- | --- | --- | --- | --- |
| LastValue | Native deterministic baseline | Same NPZ windows and stress operators | 4 src / 8 scenarios | No training; zero parameters | Latency lower bound, not a learned forecaster |
| DLinear | Native supervised baseline | Same NPZ windows, stress injection, and capacity evaluator | 4 src / 8 scenarios | Shared epochs, batch size, seed, and hardware | Low-cost learned deployment baseline |
| RACE-DLinear | Native lightweight robust baseline | Same NPZ windows and stress-aware training | 4 src / 8 scenarios | Shared budget with DLinear | Ablation vehicle; not a leading-model claim |
| PatchTST-lite | Native lightweight patch backbone | Same NPZ windows, stress injection, and latency profiler | 4 src / 8 scenarios | Shared native training budget | Patch-family robustness reference |
| Official PatchTST | Official model class in native wrapper | Native NPZ/stress pipeline when available | 2 src / 9 scenarios | Limited official run budget; reported as reference | Official patch-family comparability check |
| Native iTransformer | Official model class in native wrapper | Native NPZ/stress wrapper for selected core scenarios | 2 src / 6 scenarios | Limited official run budget; reported as reference | Transformer reference under comparable wrapper |
| Official TimeMixer | THUML Time-Series-Library model in native wrapper | Native NPZ/stress wrapper for core scenarios | 2 src / 6 scenarios | Limited official run budget; reported as reference | Multiscale-mixing reference for model-pool coverage |
| Chronos-Bolt reference | chronos-forecasting zero-shot pipeline | Target-metric zero-shot subset; no fine-tuning | 2 src / 3 scenarios | 512-window feasibility reference | Foundation-model latency and robustness context |
| iTransformer bridge | Official LTSF CSV bridge | Flattened entity-metric channels; reference only | 3 src / 4 scenarios | Separate LTSF-style setup | Not used for core native ranking |
