# StressRoute

StressRoute is the deployment-aware model-selection component of AIOpsStressBench. It turns benchmark observations into a reproducible policy for choosing a forecasting model under telemetry stress and latency constraints.

## Goal

StressRoute should answer:

```text
Given telemetry health and a deployment objective, which forecaster should run?
```

It is not a new forecasting backbone. It is a lightweight deployment policy that selects among existing models.

## Candidate Models

Core candidates:

- LastValue
- DLinear
- RACE-DLinear
- PatchTST-lite

Official/reference candidates:

- official PatchTST native
- native iTransformer where protocol is comparable

LTSF-bridge iTransformer must not drive the core routing evaluation.

## Telemetry Health Features

For each input window:

- `missing_point_ratio`
- `missing_channel_ratio`
- `delayed_tail_ratio`
- `spike_score`
- `level_shift_score`
- `volatility_score`
- `recent_trend_score`
- `latency_budget_ms`
- `deployment_objective`

## Policy v1: Interpretable Rules

Default decision rules:

| Condition | Selected model |
|---|---|
| strict latency budget | DLinear |
| high missing-channel ratio | PatchTST-lite, unless latency budget is strict |
| delayed tail | validation delayed-tail winner |
| burst or level shift | validation stress-family winner |
| low stress and capacity objective | validation capacity-cost winner |
| unknown severe stress | lowest validation worst-case capacity-risk model |

The policy must log the selected model and the reason.

## Current v1 Result Snapshot

Alibaba 2018 has a completed v1 run with DLinear, RACE-DLinear, and PatchTST-lite:

```text
outputs/stressroute_v1_alibaba_patchtst_report.csv
outputs/stressroute_v1_alibaba_patchtst_selection_report.csv
outputs/paper_tables/table_stressroute_v1.csv
paper/tables/stressroute.tex
```

Observed behavior:

- With a strict 0.2 ms latency budget, StressRoute selects DLinear.
- With a 0.5 ms budget, it can select RACE-DLinear on clean, burst, and level-shift windows.
- With a 1.0 ms or no latency budget, it selects PatchTST-lite for clean, burst, delayed-tail, point-missing, and missing-variable windows.
- PatchTST-lite reduces capacity cost in several stressed settings, but increases P95 latency by about 5.9x relative to DLinear in this run.

Interpretation: v1 is promising as a deployment policy because it converts the benchmark's tradeoff analysis into an auditable selection rule. It should still be treated as an ICDM 2027 extension until it is evaluated on at least one additional workload/resource dataset.

NetMan now has the same v1 run:

```text
outputs/stressroute_v1_netman_patchtst_report.csv
outputs/stressroute_v1_netman_patchtst_selection_report.csv
```

Observed behavior:

- Strict latency still routes to DLinear.
- Under some stresses, the capacity objective selects RACE-DLinear while the MSE objective selects PatchTST-lite.
- In NetMan burst stress, PatchTST-lite substantially lowers MSE but increases capacity cost, which supports the paper's claim that clean or raw accuracy does not fully determine deployment utility.

Interpretation: NetMan strengthens the system-evaluation story because StressRoute exposes objective-dependent model choice, not merely latency-dependent model choice.

## Policy v2: Lightweight Learned Router

v2 is now implemented as a lightweight Logistic Regression router. It is intentionally simple: the contribution is the deployment policy and evaluation protocol, not a heavy meta-model.

Current implementation:

```text
scripts/run_stressroute_v2.py
outputs/stressroute_v2_alibaba_mixed.csv
outputs/stressroute_v2_salesforce_borg_mixed.csv
outputs/stressroute_v2_alibaba_mixed_selection.csv
outputs/stressroute_v2_salesforce_borg_mixed_selection.csv
outputs/paper_tables/table_stressroute_v2.csv
outputs/paper_tables/table_stressroute_policy_summary.csv
paper/tables/stressroute_v2.tex
paper/tables/stressroute_policy.tex
```

Inputs:

- telemetry-stress features: missing-point ratio, missing-channel ratio, delay steps, noise level, burst/level-shift indicators;
- recent-window features: volatility, trend score, absolute level, missingness;
- deployment controls: latency budget and objective.

Candidates:

- DLinear
- RACE-DLinear
- PatchTST-lite

Targets:

- best model by capacity cost;
- best model by MSE.

Reference policies:

- fixed DLinear;
- fixed PatchTST-lite;
- StressRoute v1;
- StressRoute v2;
- oracle selector upper bound.

Current interpretation:

- Under a strict 0.2 ms latency budget, v1 and v2 fall back to DLinear on both Alibaba and Salesforce/Borg.
- Under a 0.5 ms budget, fixed PatchTST-lite is often lower-cost but infeasible; v1/v2 select feasible intermediate routes such as RACE-DLinear or mixed DLinear/RACE-DLinear.
- Under a 1.0 ms budget, v1 often chooses PatchTST-lite, while v2 may choose PatchTST-lite, RACE-DLinear, or mixed per-window routes depending on source and stress.
- v2 reports budget feasibility, latency versus PatchTST-lite, latency-constrained regret, and oracle gap.
- The oracle gap remains large, so v2 should be presented as a deployment-policy prototype and not as a finished routing algorithm.
- The main paper should use the compact policy summary table. The full v2 table is retained for audit in the artifact because it is too large for the main narrative.

Run commands:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> .conda-env/bin/python scripts/run_stressroute_v2.py \
  --base-config configs/alibaba2018_machine_usage.yaml \
  --source alibaba2018 \
  --dataset alibaba2018 \
  --models dlinear race_dlinear patchtst \
  --routable-models dlinear race_dlinear patchtst \
  --scenarios clean missing_30 missing_variables_30 delayed_12 burst level_shift \
  --latency-budgets 0.2 0.5 1.0 none \
  --objectives capacity mse \
  --output outputs/stressroute_v2_alibaba_mixed.csv \
  --selection-output outputs/stressroute_v2_alibaba_mixed_selection.csv \
  --device cuda

CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> .conda-env/bin/python scripts/run_stressroute_v2.py \
  --base-config configs/salesforce_borg_256x2048.yaml \
  --source salesforce_borg \
  --dataset salesforce_borg_256x2048 \
  --models dlinear race_dlinear patchtst \
  --routable-models dlinear race_dlinear patchtst \
  --scenarios clean missing_30 missing_variables_30 delayed_12 burst level_shift \
  --latency-budgets 0.2 0.5 1.0 none \
  --objectives capacity mse \
  --output outputs/stressroute_v2_salesforce_borg_mixed.csv \
  --selection-output outputs/stressroute_v2_salesforce_borg_mixed_selection.csv \
  --device cuda
```

## Required Metrics

StressRoute evaluation must report:

- MSE
- MAE
- total normalized capacity cost
- under-provision area
- peak miss rate
- P95 latency
- selected-model distribution
- oracle gap
- latency-constrained regret

## Success Criteria

StressRoute should be considered useful if it:

- lowers capacity cost or missing-variable risk relative to DLinear,
- lowers latency relative to always using PatchTST-lite,
- provides clear deployment lessons even when learned routing is unstable.

Approaching the oracle selector is desirable but not required for the main paper claim. The current v2 result is a partial success: it improves the policy story and exposes oracle headroom, but it should not be oversold as a final router. The safer main claim is that deployment-stress evidence can be converted into an auditable routing policy whose choices change with stress and latency budget.

## ICDM 2027 Positioning

StressRoute is a policy-analysis component that helps upgrade AIOpsStressBench from a table-heavy benchmark report to an Applied Track system evaluation. It should not displace the main contribution: the benchmark protocol, deployment-aware metrics, multi-source findings, and reproducible artifact.

