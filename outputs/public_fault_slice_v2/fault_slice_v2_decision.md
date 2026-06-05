# Public Fault-Injection Slice v2 Decision

- Scope: public fault-injection slice sanity evidence, not production incident validation.
- Source: RE2-OB PyG public Online Boutique fault-injection telemetry.
- Window protocol: input_len=24, pred_len=6; short-window sanity probe because RE2-OB normal/anomaly halves have 72 snapshots.
- Random seed: 42.
- Target metric: index 3 (istio_latency_99), treated as latency; the asymmetric proxy is therefore reported as decision cost rather than resource capacity cost for this probe.
- Training granularity: per_case. The forecasters are trained separately for each case rather than pooling train-normal windows across cases.
- Normal holdout: the pre-injection segment is split into train-normal and held-out-normal with normal_train_ratio=0.5; held-out-normal starts one prediction horizon after the train-normal boundary.
- Training/scaling scope: models and scalers use only train-normal windows. Held-out-normal and fault windows are never used for fitting.
- Evaluation scope: normal evaluation windows are held-out pre-injection windows; fault evaluation windows start after the injection boundary and do not cross it.
- Window counts: median train windows per case=70, median held-out normal windows per case=10, median fault windows per case=430. These counts are entity/service-expanded forecasting windows, not only temporal sliding windows, which is why the median fault-window count can exceed the roughly 72 temporal snapshots in each RE2-OB segment.
- Case scope: 58 of 60 available records are evaluated because this bounded probe uses max_cases=60; skipped records are reported separately.
- Ratio aggregation: fault/normal MSE ratio and decision-cost ratio are computed per case from the best fault-window winner divided by the matched held-out-normal winner; compact tables report medians across cases, not ratios of global means.

## Decision

Go: short main-text sentence or tiny table is supportable.

## Summary

- Parsed cases: 58
- Metric rows: 348
- Normal-to-fault MSE ranking changes: 9
- Fault-window MSE-vs-decision-cost disagreements: 11
- Skipped cases: 2

## Paper Use

Use only as public fault-injection slice sanity evidence. Do not call it production incident validation or a replacement for the controlled stress benchmark.
