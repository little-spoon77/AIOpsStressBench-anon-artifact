# Deployment Stress Protocol

AIOpsStressBench evaluates forecasting under degraded telemetry. Stress is applied only to the input window; ground-truth targets remain clean.

## Default Forecast Task

```text
input_len: 96
pred_len: 24
split: chronological 65/15/20
target_metric: 0
main_seed: 42
multi_seed: 42, 2025, 2026
```

## Stress Families

| Stress | Parameters | Operational motivation |
|---|---|---|
| clean | none | normal offline forecasting protocol |
| missing points | rate 0.10 / 0.30 / 0.50 / 0.70 | packet loss, scraper gaps, intermittent telemetry drops |
| metric outage | channel rate 0.10 / 0.30 / 0.50 | exporter failure, missing metric family, telemetry outage during incidents |
| delayed tail | k = 6 / 12 / 24 recent steps hidden | ingestion lag, late-arriving monitoring data |
| noise | std 0.10 / 0.20 / 0.40 | sensor or aggregation instability |
| burst | spike rate 0.02 | workload spike, flash crowd, transient incident |
| level shift | shift 0.40 on second half of input | release, migration, mitigation, regime change |

## Evaluation Principle

Stress operators should be deterministic under a fixed seed. For each scenario:

- Train-time stress may be enabled for native robust baselines.
- Evaluation stress is always applied to the input window only.
- Targets remain clean.
- Report accuracy, latency, memory, and forecast-to-capacity decision proxy.

## Required Scenario Sets

Core paper set:

```text
clean
missing_30
missing_variables_30
delayed_12
burst
level_shift
```

Severity set:

```text
missing points: 0, 10, 30, 50, 70%
metric outage: 0, 10, 30, 50%
delayed tail: 0, 6, 12, 24
noise: 0, 0.1, 0.2, 0.4
```

## StressRoute Set

StressRoute should be tested on mixed deployment workloads containing:

- clean
- missing_30
- missing_variables_30
- delayed_12
- burst
- level_shift

Each route decision must log:

- stress family
- telemetry health features
- selected model
- latency budget
- deployment objective
- achieved MSE / MAE / capacity cost
- oracle best model for comparison

## Non-Claims

The protocol is a reproducible stress benchmark, not a complete production incident simulator.
