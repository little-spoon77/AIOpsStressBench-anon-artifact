# Anonymous Artifact Guide

This guide is the reviewer-facing entry point for the anonymous artifact of:

`AIOpsStressBench: An Offline Stress-to-Decision Benchmark for Operational Time-Series Forecasting`

It avoids personal identifiers, host aliases, and machine-specific paths. Internal command logs remain in `EXPERIMENTS.md`.

## 1. What This Artifact Reproduces

The artifact reproduces:

- public-data conversion into the common NPZ tensor format;
- deployment-stress forecasting runs;
- generated paper tables and figures;
- compact main-paper evidence tables for cross-source winners, capacity sensitivity, and RE2-OB sanity checking;
- forecast-to-capacity proxy outputs;
- StressRoute policy summaries;
- bounded Chronos-Bolt zero-shot reference outputs;
- stress realism audit over public telemetry proxies;
- imputation plus forecasting pipeline baselines.
- bounded learned mask-aware imputation pipeline probe.
- defensive strong-probe summaries for expanded multi-seed official-model stability and StressRoute regret/oracle-gap analysis.

It does not claim to reproduce an autoscaling system, service-level SLO validation, or a complete foundation-model benchmark.
Main-paper winner tables use only the core learned comparable pool. LastValue, Chronos-Bolt, and LTSF-bridge iTransformer are reference rows and do not drive core model-ranking claims.
Stress realism audit statistics are public-trace proxies for degradation evidence, not direct unreleased incident measurements. Imputation rows are preprocessing baselines and should be read as pipeline checks rather than new backbone models.
The learned-imputation probe is a bounded mask-aware preprocessing baseline, not a SAITS/BRITS/PyPOTS benchmark.

## 1.1 Suggested Anonymous Package Contents

For review, the anonymous package should include:

- source code, configs, and scripts needed for the minimum reproduction;
- `ANON_ARTIFACT.md`, `DATASETS.md`, `STRESS_PROTOCOL.md`, `STRESSROUTE.md`, `REVIEWER_RISKS.md`, `FINAL_CLAIM_AUDIT.md`, `SUBMISSION_CHECKLIST.md`, and `reproducibility_checklist.md`;
- generated CSV sources under `outputs/paper_tables` and figure sources under `outputs/paper_figures`;
- the paper source under `paper/`;
- data preparation scripts and public-source instructions.

It should exclude shell history, host aliases, personal identifiers, absolute workstation paths, raw third-party data that cannot be redistributed, and internal-only logs unless they have been anonymized.

`FINAL_CLAIM_AUDIT.md` is the claim-to-evidence map used before submission. It links each main paper claim to the main-text table/figure and the CSV or figure artifact that supports it.

The current main paper prioritizes compact evidence tables. `paper/tables/cross_source_winners.tex` is generated from `outputs/paper_tables/table_scenario_winners.csv`; `paper/tables/capacity_sensitivity_compact.tex` is generated from `outputs/paper_tables/table_capacity_cost_ratio_sensitivity.csv` and `outputs/paper_tables/table_capacity_horizon_sensitivity.csv`; `paper/tables/public_fault_mini.tex` is generated from `outputs/public_fault_slice_v2/fault_slice_v2_compact.csv`.

Case-study stress consequences are generated from the existing case-study metric files under `outputs/paper_figures/*_metrics.csv` and summarized in `outputs/paper_tables/table_case_consequence.csv`. They support mechanism-level checks rather than main-paper aggregate evidence.

## 2. Environment

Use a project-local Python environment:

```bash
cd <repo_root>
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

GPU runs should use an explicitly selected idle GPU:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> python <script> ...
```

The minimum reproduction can run on one GPU. Full official-baseline and sensitivity sweeps may take longer and can be distributed across multiple GPUs.

## 3. Data Format

All processed datasets use the same NPZ schema:

```text
series: float32 [entities, time, metrics]
metric_names: string [metrics]
entity_ids: string [entities]
```

Public source roles:

- Alibaba 2018: multivariate machine-level resource telemetry.
- Salesforce/Borg 2011: multivariate operational telemetry.
- GAIA: KPI stress diversity.
- NetMan: KPI telemetry and missing-telemetry case studies.

Capacity-related conclusions are limited to resource-level operational telemetry. They should not be read as service-level SLO behavior or autoscaling savings.

Large public data files should be obtained from their original sources using the provided preparation scripts. The artifact should not redistribute third-party raw data unless the source license explicitly allows it.

## 4. Minimum Reproduction

The minimum reproduction checks the pipeline without rerunning every long experiment:

```bash
cd <repo_root>
CUDA_VISIBLE_DEVICES=<idle_gpu> python -m race_forecast.run \
  --config configs/quick_synthetic.yaml

CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> python scripts/run_stress_suite.py \
  --base-config configs/alibaba2018_machine_usage.yaml \
  --output-root outputs/minimal_alibaba_stress \
  --scenarios clean missing_30 missing_variables_30 delayed_12
```

Expected outputs:

```text
outputs/minimal_alibaba_stress/<scenario>/<model>/metrics.json
outputs/minimal_alibaba_stress/<scenario>/<model>/metrics.csv
```

Stress realism audit:

```bash
PYTHONPATH=<repo_root> python scripts/audit_stress_realism.py \
  --dataset gaia=data/gaia_metric_forecast.npz \
  --dataset netman=data/netman_kpi.npz \
  --dataset alibaba2018=data/alibaba2018_machine_usage.npz \
  --dataset salesforce_borg=data/salesforce_borg_256x2048.npz \
  --output-dir outputs/paper_tables

PYTHONPATH=<repo_root> python scripts/calibrate_real_degradation.py \
  --dataset alibaba2018=data/alibaba2018_machine_usage.npz \
  --dataset salesforce_borg=data/salesforce_borg_256x2048.npz \
  --dataset gaia=data/gaia_metric_forecast.npz \
  --dataset netman=data/netman_kpi.npz \
  --output-dir outputs/real_degradation_calibration

CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> python scripts/run_natural_degradation_slices.py \
  --output-dir outputs/natural_proxy_slice \
  --paper-table-dir outputs/paper_tables \
  --latex-output paper/tables/natural_degradation_slices.tex \
  --seed 42 \
  --device auto \
  --top-fraction 0.05 \
  --min-windows 128 \
  --max-slice-windows 1000 \
  --epochs 10 \
  --batch-size 128 \
  --max-train-windows 8192 \
  --max-val-windows 2048 \
  --max-test-windows 20000
```

Imputation pipeline baselines:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> python scripts/run_imputation_pipeline.py \
  --resume \
  --datasets alibaba2018 salesforce_borg \
  --scenarios clean missing_30 missing_variables_30 delayed_12 level_shift \
  --imputations none ffill mean \
  --models dlinear patchtst \
  --output-root outputs/imputation_pipeline \
  --summary outputs/imputation_pipeline_summary.csv \
  --device cuda
```

Bounded learned mask-aware imputation probe:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> python scripts/run_learned_imputation_pipeline.py \
  --output-dir outputs/learned_imputation_pipeline \
  --seed 42 \
  --device auto \
  --imputer-epochs 4 \
  --forecaster-epochs 4 \
  --batch-size 128 \
  --max-train-windows 4096 \
  --max-val-windows 1024 \
  --max-test-windows 12000
```

## 5. Main Table and Figure Generation

Generate paper tables:

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
  --multiseed outputs/multiseed_summary.csv \
  --multiseed outputs/multiseed_salesforce_borg_256x2048_summary.csv \
  --stressroute outputs/stressroute_v1_alibaba_patchtst_report.csv \
  --stressroute outputs/stressroute_v1_salesforce_borg_256x2048_report.csv \
  --stressroute-v2 outputs/stressroute_v2_alibaba_mixed.csv \
  --stressroute-v2 outputs/stressroute_v2_salesforce_borg_mixed.csv \
  --imputation outputs/imputation_pipeline_summary.csv \
  --output-dir outputs/paper_tables

PYTHONPATH=<repo_root> python scripts/make_latex_tables.py \
  --table-dir outputs/paper_tables \
  --output-dir paper/tables

PYTHONPATH=<repo_root> python scripts/make_multisource_deployment_figure.py \
  --severity outputs/severity_curve_summary.csv outputs/severity_curve_salesforce_borg_256x2048_summary.csv \
  --capacity outputs/capacity_simulator_summary.csv outputs/capacity_simulator_salesforce_borg_256x2048_summary.csv \
  --winners outputs/paper_tables/table_scenario_winners.csv \
  --stressroute-selection outputs/stressroute_v2_alibaba_mixed_selection.csv outputs/stressroute_v2_salesforce_borg_mixed_selection.csv \
  --output-dir outputs/paper_figures
```

Compile the paper:

```bash
cd <repo_root>/paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The canonical paper PDF is `paper/main.pdf`, generated from `paper/main.tex`.

## 6. Full Reproduction Components

Full reproduction includes:

- native lightweight stress suites on Alibaba and Salesforce/Borg;
- official PatchTST native wrapper;
- native iTransformer wrapper;
- official TimeMixer native wrapper;
- bounded Chronos-Bolt zero-shot reference, including the A6000 outputs under `outputs/strong_probe/foundation_reference_a6000`;
- forecast-to-capacity decision proxy and sensitivity sweeps;
- StressRoute policy reports;
- natural proxy slice evaluation without synthetic stress injection;
- public fault-injection slice probes: RCAEval RE1-OB retained as artifact-only caution evidence, and RE2-OB Online Boutique retained as short-window public fault-injection sanity evidence;
- bounded learned mask-aware imputation probe under missing telemetry;
- defensive expanded multi-seed official-model probes;
- StressRoute regret/oracle-gap summaries derived from existing policy outputs;
- case-study overlays;
- manifest and LaTeX checks.
- case-study overlay files generated from existing case metrics.

Public fault-injection slice probes:

```bash
PYTHONPATH=<repo_root> python scripts/run_public_fault_slice_eval.py \
  --input data/raw/rcaeval/RE1-OB.zip \
  --output-dir outputs/public_fault_slice

PYTHONPATH=<repo_root> python scripts/run_public_fault_slice_eval_v2.py \
  --input data/raw/re2ob_pyg/re2ob_pyg.pkl \
  --output-dir outputs/public_fault_slice_v2
```

These probes are not incident validation. The RCAEval RE1-OB slice parses successfully, but the evaluated normal and fault windows do not show strong enough forecasting separation for a main-text claim, so it is retained as caution evidence. The RE2-OB Online Boutique slice uses public fault-injection windows with aligned normal/fault segments. The pre-injection normal segment is split into train-normal and held-out-normal with a one-horizon gap; models and scalers are fit only on train-normal windows, and fault evaluation windows start after the injection boundary. The forecasters are trained per case rather than from pooled cases with seed 42. The target metric is `istio_latency_99`, so the asymmetric proxy is reported as decision cost rather than resource-capacity cost for this probe. In 58 parsed cases out of 60 bounded records, 9 cases change the best-MSE model from held-out-normal to fault windows and 11 fault windows have different best-MSE and best-decision-cost models. Median window counts are 70 train-normal, 10 held-out-normal, and 430 fault windows per case; these are entity/service-expanded forecasting windows, not only temporal sliding windows, which explains why the median fault-window count can exceed the roughly 72 temporal snapshots in each RE2-OB segment. Median per-case fault/normal ratios are 53.34 for MSE and 1.49 for decision cost. This supports the paper's limited claim that public fault-injection slices can expose model-selection risk without synthetic corruption.

The RE2-OB v2 output files are:

```text
outputs/public_fault_slice_v2/fault_slice_v2_metrics.csv
outputs/public_fault_slice_v2/fault_slice_v2_winners.csv
outputs/public_fault_slice_v2/fault_slice_v2_summary.csv
outputs/public_fault_slice_v2/fault_slice_v2_compact.csv
outputs/public_fault_slice_v2/fault_slice_v2_decision.md
outputs/public_fault_slice_v2/fault_slice_v2_skipped.json
```

The A6000 Chronos-Bolt reference files are:

```text
outputs/strong_probe/foundation_reference_a6000/foundation_reference_summary.csv
outputs/strong_probe/foundation_reference_a6000/foundation_reference_compact_table.csv
outputs/strong_probe/foundation_reference_a6000/foundation_reference_decision.md
```

These rows are bounded zero-shot references only; they do not enter the core comparable winner pool.

The defensive strong-probe files are:

```text
outputs/strong_probe/expanded_multiseed/
outputs/strong_probe/defensive_tables/expanded_multiseed_official_stats.csv
outputs/strong_probe/defensive_tables/expanded_multiseed_official_winners.csv
outputs/strong_probe/defensive_tables/expanded_multiseed_official_decision.md
outputs/strong_probe/defensive_tables/multiseed_core_compact.csv
outputs/strong_probe/defensive_tables/stressroute_regret_compact.csv
outputs/strong_probe/defensive_tables/stressroute_regret_decision.md
```

These files are reviewer-defense artifacts. They support stability and policy-regret checks but do not replace the main-paper tables.

The natural proxy slice files are:

```text
outputs/natural_proxy_slice/natural_proxy_slice_metrics.csv
outputs/natural_proxy_slice/natural_proxy_slice_winners.csv
outputs/natural_proxy_slice/natural_proxy_slice_summary.csv
outputs/natural_proxy_slice/natural_proxy_slice_decision.md
outputs/natural_proxy_slice/natural_slice_summary.csv
outputs/natural_proxy_slice/natural_slice_winners.csv
outputs/natural_proxy_slice/natural_slice_report.md
outputs/paper_tables/table_natural_degradation_slices.csv
paper/tables/natural_degradation_slices.tex
```

These rows evaluate naturally abnormal public-trace proxy windows without synthetic stress injection and compare them with matched low-proxy normal windows from the same source. They are not labeled production incidents.

Natural proxy slice reproducibility and fairness checks:

- Salesforce/Borg all-zero `dynamic_4` is excluded from every natural proxy score by the global all-zero / near-constant metric mask.
- Matched normal windows are sampled from low-proxy test windows and exclude the union of all four proxy families' top-5% degraded candidate windows, not only the finally selected capped degraded windows.
- Each degraded slice is selected from the top-5% proxy windows and capped at `--max-slice-windows 1000`.
- Degraded and matched normal slices are drawn only from the clean test split and are used only for evaluation after training; they are not used for training, hyperparameter tuning, early stopping, or model selection.
- Capacity proxy uses the same defaults as the main experiments: headroom `h=0.15`, under-cost `c_u=5.0`, over-cost `c_o=1.0`, demand floor `epsilon=0.05`, and target metric index `0`. Accuracy and capacity use the same target.
- `outputs/paper_tables/table_natural_degradation_slices.csv` and `paper/tables/natural_degradation_slices.tex` are generated automatically by `scripts/run_natural_degradation_slices.py` using `--paper-table-dir` and `--latex-output`; they are not hand-written.

Generation command:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> PYTHONPATH=<repo_root> python scripts/run_natural_degradation_slices.py \
  --output-dir outputs/natural_proxy_slice \
  --paper-table-dir outputs/paper_tables \
  --latex-output paper/tables/natural_degradation_slices.tex \
  --seed 42 \
  --device auto \
  --top-fraction 0.05 \
  --min-windows 128 \
  --max-slice-windows 1000 \
  --epochs 10 \
  --batch-size 128 \
  --max-train-windows 8192 \
  --max-val-windows 2048 \
  --max-test-windows 20000
```

The default forecast-to-capacity protocol uses headroom `h=0.15`, under-provision cost `c_u=5.0`, over-provision cost `c_o=1.0`, demand floor `epsilon=0.05`, and target metric index `0`. The artifact also recomputes 2:1, 5:1, and 10:1 under/over cost ratios from existing simulator summaries. Reactive HPA is a controller reference, not part of the forecasting model pool.

Severity slopes are linear slopes of relative MSE against the stress level or rate. For metric outage, multiplying the slope by a missing-variable rate such as 0.5 gives the approximate relative-MSE increase visible in the severity figure.

These runs are separated from the minimum reproduction because they are more expensive and may require multiple GPUs.

## 7. Artifact Checks

Run the manifest check:

```bash
PYTHONPATH=<repo_root> python scripts/check_benchmark_manifest.py \
  --manifest benchmark_manifest.yaml \
  --root <repo_root>
```

The check verifies that required tables, figures, and reference summaries exist.

## 8. Non-Claims

The artifact does not claim:

- RACE-DLinear is a new forecasting architecture;
- no claim that the forecast-to-capacity proxy reproduces an autoscaling system;
- Alibaba or Salesforce/Borg are service-level SLO traces;
- Chronos-Bolt tiny represents all foundation models;
- StressRoute is a finished router or optimized deployment policy.

