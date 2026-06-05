# Reproducibility Checklist

## Paper Identity

- Title: `AIOpsStressBench: Deployment-Stress Benchmarking for Operational Time-Series Forecasting`
- Track positioning: ICDM Applied Track benchmark and system evaluation.
- Main contribution: deployment-stress protocol, public-data unification, accuracy/robustness/latency/capacity tradeoff analysis, and case studies.
- Non-claim: RACE-DLinear is not a SOTA method contribution.

## Data

| Dataset | File | Shape | Role | Caveat |
|---|---|---:|---|---|
| GAIA | `data/gaia_metric_forecast.npz` | `[127, 4092, 1]` | KPI stress diversity | mostly single-target KPI series |
| NetMan | `data/netman_kpi.npz` | `[29, 4096, 1]` | telemetry-outage case studies | KPI semantics are not resource capacity |
| Alibaba2018 | `data/alibaba2018_machine_usage.npz` | `[128, 4096, 5]` | main multivariate resource telemetry | machine-level, not service-level |
| SalesforceBorg | `data/salesforce_borg_256x2048.npz` | `[256, 2048, 7]` | second multivariate operational telemetry source | one dynamic metric is all zero in the current subset |

NPZ schema:

```text
series: float32 [entities, time, metrics]
metric_names: string [metrics]
entity_ids: string [entities]
```

Public-data acquisition attempts that did not become main results are recorded in `data_source_attempts.md`.

## Forecast Task

- Input length: 96
- Prediction horizon: 24 for main tables
- Horizon sensitivity: 12, 24, 48, 96
- Split: chronological
- Train/validation/test ratio: 65/15/20
- Main seed: 42
- Multi-seed stability: 42, 2025, 2026
- Target policy: stress is applied only to the input window; ground-truth targets remain clean.

## Stress Protocol

| Name | Definition |
|---|---|
| clean | no injected corruption |
| missing_10 / 30 / 50 / 70 | random point missingness at the specified rate |
| missing_variables_10 / 30 / 50 | entire metric-channel outage at the specified rate |
| delayed_6 / 12 / 24 | last k input steps unavailable |
| noise_10 / noisy / noise_40 | Gaussian input noise with std 0.10 / 0.20 / 0.40 |
| burst | sparse positive spikes with rate 0.02 |
| level_shift | add 0.40 to the second half of the input window |

## Baselines

Native baselines:

- LastValue
- DLinear
- RACE-DLinear
- PatchTST-lite

Official baselines:

- Official PatchTST through native NPZ/stress pipeline.
- Official iTransformer through native wrapper for selected core scenarios.
- Official TimeMixer through THUML Time-Series-Library in the native wrapper.
- Chronos-Bolt through a bounded target-metric zero-shot reference run.
- Older official iTransformer LTSF bridge retained as a reference, with protocol limitation stated.

## Metrics

Forecast quality:

- MSE
- MAE

Deployment cost:

- P50 latency
- P95 latency
- parameter count
- GPU memory
- train/evaluation time where available

Forecast-to-capacity simulator:

- default headroom: 0.15
- under-provision cost: 5.0
- over-provision cost: 1.0
- cost-ratio sensitivity: 2:1, 5:1, 10:1
- demand floor / epsilon: 0.05
- target metric index: 0
- under-provision rate
- under-provision area
- over-provision area
- peak miss rate
- severe-under rate
- P95 under-provision ratio
- total normalized cost

The simulator is a deployment proxy, not a production autoscaling simulator.
Reactive HPA is a controller reference, not a forecasting model pool member.

Main winner tables use the core learned comparable pool only. LastValue, Chronos-Bolt, and LTSF-bridge iTransformer are retained as lower-bound or reference rows but are not eligible for central winner claims.

Severity slopes are computed as linear slopes of relative MSE versus stress level/rate. The compact multi-seed table in the main text is a subset of `table_multiseed_stability.csv`.

Stress realism audit is generated from public NPZ tensors and reports observable degradation proxies, not unreleased operator incident distributions.

Imputation pipeline baselines use the same native stress protocol with forward-fill and mean-fill preprocessing after stress injection.

Chronos-Bolt A6000 bounded-reference outputs are stored under:

```text
outputs/strong_probe/foundation_reference_a6000/foundation_reference_summary.csv
outputs/strong_probe/foundation_reference_a6000/foundation_reference_compact_table.csv
outputs/strong_probe/foundation_reference_a6000/foundation_reference_decision.md
```

## Artifact Files

- Manifest: `benchmark_manifest.yaml`
- Experiment notes: `EXPERIMENTS.md`
- Artifact guide: `ARTIFACT.md`
- Anonymous artifact entry point: `ANON_ARTIFACT.md`
- Claim audit: `FINAL_CLAIM_AUDIT.md`
- Data-source attempts: `data_source_attempts.md`
- Main table CSV/Markdown: `outputs/paper_tables/`
- LaTeX table snippets: `paper/tables/`
- Figures: `outputs/paper_figures/`
- Draft: `paper/main.tex`
- PDF: `paper/main.pdf`

## Generation Commands

Core tables:

```bash
PYTHONPATH=<repo_root> python scripts/make_paper_tables.py \
  --dataset GAIA=data/gaia_metric_forecast.npz \
  --dataset NetMan=data/netman_kpi.npz \
  --dataset Alibaba2018=data/alibaba2018_machine_usage.npz \
  --dataset SalesforceBorg=data/salesforce_borg_256x2048.npz \
  --metrics-summary salesforce_borg=outputs/salesforce_borg_256x2048_stress_summary.csv \
  --official-patchtst outputs/official_patchtst_native_summary.csv \
  --official-patchtst outputs/official_patchtst_salesforce_borg_256x2048_summary.csv \
  --official-itransformer-native outputs/official_itransformer_native_summary.csv \
  --official-itransformer-native outputs/official_itransformer_salesforce_borg_256x2048_summary.csv \
  --official-timemixer outputs/official_timemixer_native_summary.csv \
  --official-timemixer outputs/official_timemixer_salesforce_borg_256x2048_summary.csv \
  --chronos-reference outputs/chronos_reference_alibaba2018_summary.csv \
  --chronos-reference outputs/chronos_reference_salesforce_borg_256x2048_summary.csv \
  --capacity-simulator outputs/capacity_simulator_summary.csv \
  --capacity-simulator outputs/capacity_simulator_salesforce_borg_256x2048_summary.csv \
  --capacity-simulator-sensitivity outputs/capacity_simulator_sensitivity_summary.csv \
  --capacity-simulator-sensitivity outputs/capacity_simulator_salesforce_borg_256x2048_summary.csv \
  --severity outputs/severity_curve_summary.csv \
  --severity outputs/severity_curve_salesforce_borg_256x2048_summary.csv \
  --official-severity outputs/official_patchtst_severity_summary.csv \
  --multiseed outputs/multiseed_summary.csv \
  --multiseed outputs/multiseed_salesforce_borg_256x2048_summary.csv \
  --stressroute outputs/stressroute_v1_alibaba_patchtst_report.csv \
  --stressroute outputs/stressroute_v1_salesforce_borg_256x2048_report.csv \
  --stressroute-v2 outputs/stressroute_v2_alibaba.csv \
  --stressroute-v2 outputs/stressroute_v2_salesforce_borg.csv \
  --output-dir outputs/paper_tables
```

LaTeX tables:

```bash
PYTHONPATH=<repo_root> python scripts/make_latex_tables.py \
  --table-dir outputs/paper_tables \
  --output-dir paper/tables
```

Multi-source deployment-stress figure:

```bash
PYTHONPATH=<repo_root> python scripts/make_multisource_deployment_figure.py \
  --severity outputs/severity_curve_summary.csv outputs/severity_curve_salesforce_borg_256x2048_summary.csv \
  --capacity outputs/capacity_simulator_summary.csv outputs/capacity_simulator_salesforce_borg_256x2048_summary.csv \
  --stressroute-selection outputs/stressroute_v2_alibaba_selection.csv outputs/stressroute_v2_salesforce_borg_selection.csv \
  --output-dir outputs/paper_figures
```

Paper compilation:

```bash
cd <repo_root>/paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Optional official PatchTST NetMan supplement:

```bash
GPU_ID=<idle_gpu> MAX_USED_MB=1024 MAX_UTIL=20 MAX_CHECKS=1 \
  bash scripts/run_patchtst_netman_when_free.sh
```

StressRoute v1 smoke test:

```bash
PYTHONPATH=<repo_root> python scripts/run_stressroute_v1.py \
  --base-config configs/quick_synthetic.yaml \
  --source synthetic \
  --dataset synthetic \
  --models last_value dlinear race_dlinear \
  --routable-models dlinear race_dlinear \
  --scenarios clean missing_30 missing_variables_30 \
  --latency-budgets 0.2 none \
  --objectives capacity mse \
  --output outputs/stressroute_v1_smoke.csv \
  --selection-output outputs/stressroute_v1_smoke_selection.csv \
  --device cpu
```

StressRoute v2 on the two multivariate telemetry sources:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> python scripts/run_stressroute_v2.py \
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

CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> python scripts/run_stressroute_v2.py \
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

Artifact consistency check:

```bash
PYTHONPATH=<repo_root> python scripts/check_benchmark_manifest.py \
  --manifest benchmark_manifest.yaml \
  --root <repo_root>
```

## Hardware and Runtime Notes

- Execution root: `<repo_root>`
- GPU type observed: NVIDIA GeForce RTX 4090
- Runs should set `CUDA_VISIBLE_DEVICES=<idle_gpu>`.
- Always inspect `nvidia-smi` before launching a job.
- Do not kill existing jobs.

## Reviewer-Risk Controls

- The paper states that Alibaba is machine-level telemetry, not service-level request telemetry.
- The paper states that GAIA/NetMan support KPI stress diversity, not strong capacity-planning claims.
- The paper states that the capacity simulator is a proxy.
- The paper states that official iTransformer LTSF bridge is not perfectly comparable to native NPZ stress.
- The paper states that Chronos-Bolt is only a bounded zero-shot reference, not a complete foundation-model comparison.
- The paper avoids claiming RACE-DLinear as SOTA.


