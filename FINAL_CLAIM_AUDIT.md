# Final Claim Audit

This audit maps the paper's main claims to visible paper evidence and artifact files.
It is intended for final submission checking, not as a paper section.

## Claim 1: Clean evaluation can change within-protocol model selection

- Paper evidence:
  - `paper/main.tex`: Abstract, Introduction, Claim 1, Conclusion.
  - `paper/tables/cross_source_winners.tex`
  - `paper/tables/clean_alibaba.tex`
  - `paper/tables/scenario_winners.tex`
  - `paper/tables/deployment_cost.tex`
- Artifact sources:
  - `outputs/paper_tables/table_clean_accuracy.csv`
  - `outputs/paper_tables/table_scenario_winners.csv`
  - `outputs/paper_tables/table_deployment_cost.csv`
- Scope:
  - Winner claims use the core learned comparable pool.
  - LastValue, Chronos-Bolt, and LTSF-bridge iTransformer are not part of the core winner table.
  - The main paper now opens Results with a cross-source compact winner table over Alibaba and Salesforce/Borg, so the first evidence is not Alibaba-only.

## Claim 2: Robustness is stress-family specific

- Paper evidence:
  - `paper/main.tex`: Claim 2 and Stability Across Sources.
  - `paper/main.tex`: Figure 2 caption and severity/disagreement panels.
- Artifact sources:
  - `outputs/paper_tables/table_structural_findings.csv`
  - `outputs/paper_tables/table_severity_slope.csv`
  - `outputs/paper_tables/table_severity_auc.csv`
  - `outputs/paper_figures/figure_multisource_deployment_stress.csv`
  - `outputs/paper_figures/figure_multisource_winner_disagreement.csv`
- Scope:
  - The source-dependent metric-outage slopes are auxiliary severity evidence, not the sole flagship claim.
  - The main stability evidence is the five-seed source-stress analysis from `outputs/five_seed`: best MSE and best decision-cost winners differ in 4/10 core settings, and paired tests over decision cost give p < 0.01 for all four flips.
  - Noise slopes should be interpreted as smaller absolute degradation and smaller model-to-model gap, not as universal robustness.

## Claim 3: Stress operators are operationally motivated, not arbitrary corruption

- Paper evidence:
  - `paper/tables/stress_taxonomy.tex`
  - `paper/tables/public_fault_mini.tex`
  - `paper/main.tex`: Stress Realism Audit.
- Artifact sources:
  - `outputs/paper_tables/table_stress_realism_audit.csv`
  - `outputs/natural_proxy_slice/natural_proxy_slice_metrics.csv`
  - `outputs/natural_proxy_slice/natural_proxy_slice_winners.csv`
  - `outputs/natural_proxy_slice/natural_proxy_slice_summary.csv`
  - `outputs/natural_proxy_slice/natural_proxy_slice_decision.md`
  - `outputs/natural_proxy_slice/natural_slice_report.md`
  - `outputs/paper_tables/table_natural_degradation_slices.csv`
  - `paper/tables/natural_degradation_slices.tex`
  - `outputs/public_fault_slice/fault_slice_decision.md`
  - `outputs/public_fault_slice_v2/fault_slice_v2_metrics.csv`
  - `outputs/public_fault_slice_v2/fault_slice_v2_winners.csv`
  - `outputs/public_fault_slice_v2/fault_slice_v2_summary.csv`
  - `outputs/public_fault_slice_v2/fault_slice_v2_compact.csv`
  - `outputs/public_fault_slice_v2/fault_slice_v2_decision.md`
- Scope:
  - Public traces are cleaned and underrepresent monitoring failures.
  - Tail-flatline and flatline statistics are proxies, not direct evidence of real ingestion delay.
  - Natural proxy slice rows are reviewer-defense / artifact evidence, not main-claim evidence. They are sanity checks on naturally abnormal public-trace windows. Alibaba shows MSE and decision-objective changes, while Salesforce/Borg mainly shows a capacity-winner change.
  - Public fault-injection slice evaluation is a sanity check, not incident validation. RCAEval RE1-OB is artifact-only caution evidence because normal and fault windows do not show strong enough forecasting separation. RE2-OB Online Boutique is now summarized in the main paper as a compact sanity-check row: the pre-injection segment is split into train-normal and held-out-normal with a one-horizon gap; models and scalers use only train-normal windows; fault evaluation starts after the injection boundary; forecasters are trained per case with seed 42; and 58 parsed cases produce 9 held-out-normal-to-fault best-MSE changes plus 11 fault-window MSE-vs-decision-cost disagreements. The target is Istio P99 latency, so the proxy is labeled decision cost rather than resource-capacity cost for this probe.
  - Documentation citations support the operational plausibility of missing data, stale series, metric latency, and unavailable autoscaling metrics.

## Claim 3a: AIOpsStressBench is not only another clean forecasting benchmark

- Paper evidence:
  - `paper/main.tex`: Related Work.
- Artifact sources:
  - `ANON_ARTIFACT.md`
  - `benchmark_manifest.yaml`
- Scope:
  - The positioning claim is qualitative: AIOpsStressBench combines operational telemetry, controlled deployment stress, latency/memory, forecast-to-capacity decision proxy, policy illustration, and artifact support.
  - It does not claim that earlier benchmarks are invalid; it states that they do not jointly cover the stress-to-decision evaluation target.

## Claim 4: Imputation helps point missingness but does not eliminate outage/stale-context risk

- Paper evidence:
  - `paper/tables/imputation_pipeline.tex`
  - `paper/main.tex`: Imputation Pipeline section.
- Artifact sources:
  - `outputs/paper_tables/table_imputation_pipeline.csv`
  - `outputs/learned_imputation_pipeline/learned_imputation_metrics.csv`
  - `outputs/learned_imputation_pipeline/learned_imputation_summary.csv`
  - `outputs/learned_imputation_pipeline/learned_imputation_decision.md`
- Scope:
  - Imputation rows are pipeline baselines.
  - They do not enter the core scenario-winner table unless explicitly labeled as pipeline comparisons.
  - The learned mask-aware imputer is a bounded preprocessing probe, not a complete SAITS/BRITS/PyPOTS comparison.
  - Its role is to test whether learned preprocessing changes pipeline choice under stress; it should not be described as solving telemetry outage.

## Claim 5: Forecast-to-capacity decision proxy adds information beyond MSE/MAE

- Paper evidence:
  - `paper/main.tex`: Metrics and Forecast-to-Capacity Decision Proxy.
  - `paper/tables/capacity_simulator.tex`
  - `paper/tables/scenario_winners.tex`
  - `paper/tables/capacity_sensitivity_compact.tex`
  - `paper/tables/cross_source_winners.tex`
- Artifact sources:
  - `outputs/paper_tables/table_capacity_simulator.csv`
  - `outputs/paper_tables/table_capacity_simulator_winners.csv`
  - `outputs/paper_tables/table_capacity_cost_ratio_sensitivity.csv`
  - `outputs/paper_tables/table_capacity_headroom_sensitivity.csv`
  - `outputs/paper_tables/table_capacity_horizon_sensitivity.csv`
- Scope:
  - The forecast-to-capacity evaluator is a decision proxy, not an autoscaling system or savings claim.
  - Reactive HPA is a controller reference and not eligible for forecasting winner selection.
  - Capacity conclusions are limited to resource-level operational telemetry.
  - Main-paper sensitivity is compact and metric-outage focused: `table_capacity_cost_ratio_sensitivity.csv` supports 2:1, 5:1, and 10:1 rows, while `table_capacity_horizon_sensitivity.csv` supports the horizon-96 row. PatchTST-lite is the lowest total-cost forecasting policy on both multivariate sources for those metric-outage settings.

## Claim 6: Findings reproduce across two multivariate operational telemetry sources

- Paper evidence:
  - `paper/main.tex`: Dataset section and Stability Across Sources.
  - `paper/tables/cross_source_winners.tex`
  - `paper/tables/multiseed_compact.tex`
  - `paper/tables/capacity_simulator.tex`
  - `paper/main.tex`: Figure 2.
- Artifact sources:
  - `outputs/paper_tables/table_dataset_audit.csv`
  - `outputs/paper_tables/table_scenario_winners.csv`
  - `outputs/paper_tables/table_multiseed_compact.csv`
  - `outputs/paper_tables/table_multiseed_stability.csv`
  - `outputs/paper_figures/figure_multisource_deployment_stress.csv`
  - `outputs/paper_figures/figure_multisource_winner_disagreement.csv`
  - `outputs/strong_probe/defensive_tables/multiseed_core_compact.csv`
  - `outputs/strong_probe/defensive_tables/expanded_multiseed_official_stats.csv`
  - `outputs/strong_probe/defensive_tables/expanded_multiseed_official_winners.csv`
  - `outputs/strong_probe/defensive_tables/expanded_multiseed_official_decision.md`
- Scope:
  - Alibaba and Salesforce/Borg are resource-level operational telemetry.
  - GAIA and NetMan support KPI stress diversity and case studies, not strong resource-capacity claims.
  - The strong-probe official-model multi-seed files are artifact-level defense checks. They are not required for the main-paper winner table.

## Claim 6a: Case studies illustrate stress mechanisms

- Paper evidence:
  - `paper/main.tex`: Case Studies.
- Artifact sources:
  - `outputs/paper_tables/table_case_consequence.csv`
  - `outputs/paper_figures/*_metrics.csv`
  - `outputs/paper_figures/*_overlay.pdf`
- Scope:
  - The main paper keeps a compact case-study paragraph and moves overlay figures out of the main text to make room for cross-source, capacity-sensitivity, and RE2-OB evidence.
  - Case metrics are derived from existing case-study windows.
  - They summarize local capacity proxy, under-provision rate, and P95 latency; they are not new experiments.

## Claim 7: Chronos-Bolt is a bounded foundation-model reference, not a core baseline

- Paper evidence:
  - `paper/main.tex`: Related Work, Baselines, Claim 2, Reproducibility.
- Artifact sources:
  - `outputs/strong_probe/foundation_reference_a6000/foundation_reference_summary.csv`
  - `outputs/strong_probe/foundation_reference_a6000/foundation_reference_compact_table.csv`
  - `outputs/strong_probe/foundation_reference_a6000/foundation_reference_decision.md`
  - `outputs/strong_probe/foundation_reference_a6000/foundation_reference_failures.md`
- Scope:
  - Chronos-Bolt tiny/small/base were evaluated zero-shot on A6000 GPU1.
  - Chronos-Bolt is not included in the core comparable winner pool.
  - TimesFM, Moirai, fine-tuned foundation-model variants, and probabilistic FM-specific evaluations are outside the current benchmark scope.

## Claim 8: StressRoute is a decision-support illustration, not a finished routing algorithm

- Paper evidence:
  - `paper/main.tex`: Claim 3b / model-selection policy section.
  - `paper/tables/practitioner_decision_box.tex`
  - `paper/tables/stressroute_policy.tex`
- Artifact sources:
  - `outputs/paper_tables/table_stressroute_policy_summary.csv`
  - `outputs/paper_tables/table_stressroute_v1.csv`
  - `outputs/paper_tables/table_stressroute_v2.csv`
  - `outputs/strong_probe/defensive_tables/stressroute_regret_compact.csv`
  - `outputs/strong_probe/defensive_tables/stressroute_regret_summary.csv`
  - `outputs/strong_probe/defensive_tables/stressroute_selection_distribution.csv`
  - `outputs/strong_probe/defensive_tables/stressroute_regret_decision.md`
- Scope:
  - StressRoute shows that benchmark evidence can instantiate auditable model selection.
  - It should not be presented as a mature router or optimized deployment policy.
  - The regret/oracle-gap probe compares only latency-feasible fixed baselines in the compact table.

## Final Submission Checks

- Main paper does not claim autoscaling savings.
- Main paper does not claim service-level SLO validation.
- RACE-DLinear is not a contribution bullet or new-model claim.
- Chronos-Bolt is not part of core winner selection.
- All main tables and figures have CSV or generated artifact sources.
