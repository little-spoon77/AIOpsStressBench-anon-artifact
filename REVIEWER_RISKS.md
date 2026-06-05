# Reviewer Risks and Responses

This document records the main ICDM Applied Track review risks for AIOpsStressBench and how the current artifact responds to them.

## Risk 1: This is only a normal forecasting benchmark

Reviewer concern:

- The paper may look like several existing models evaluated on several datasets.
- The statement "clean MSE is not enough" may be considered obvious.

Response:

- The main contribution is not a new backbone. It is a controlled offline deployment-stress benchmark and system evaluation.
- The protocol covers missing points, metric outage, delayed telemetry, noise, burst, and level shift.
- The evaluation reports accuracy, latency, memory, forecast-to-capacity decision-proxy cost, severity curves, multi-seed stability, case studies, and StressRoute policy behavior.
- The paper now uses a generated Benchmark Card, a compact baseline-fairness paragraph, and a cross-source winner table so the novelty is not only "many experiments" but a reproducible offline deployment-stress protocol plus quantified objective-disagreement lessons.
- The main paper now starts Results with a cross-source winner table over Alibaba and Salesforce/Borg, so the central evidence is not Alibaba-only.
- The strongest novelty claim is stress-family structure: patch-based context helps most under metric outage, while noise stress shows much smaller model-family separation and lower absolute degradation.
- The main paper now includes a compact five-seed stability table for the lightweight core pool across 10 source-stress settings, including aggregate MSE-vs-decision reversals.

Rebuttal-ready answer:

The paper is not positioned as another clean forecasting leaderboard. It contributes a reproducible offline stress-to-decision protocol and shows, first in a cross-source winner table, that clean/stressed MSE, latency, and decision-cost objectives can select different models. The severity curves and objective-disagreement panel then explain which stress families drive the disagreement.

## Risk 2: CloudOps data realism is insufficient

Reviewer concern:

- Alibaba 2018 is machine-level telemetry, not service-level request/SLO data.
- GAIA and NetMan are KPI datasets and should not support strong capacity-planning claims.

Response:

- Alibaba 2018 and Salesforce/Borg 2011 are the two multivariate operational telemetry sources used for the strongest resource-level claims.
- GAIA and NetMan are explicitly positioned as KPI stress diversity and failure-case evidence.
- The paper states that the forecast-to-capacity evaluator is a decision proxy and does not claim autoscaling savings.
- The Salesforce/Borg subset is publicly acquired and converted through documented scripts.
- Capacity-related conclusions are scoped to resource-level telemetry and are not extrapolated to service-level SLO behavior.
- The stress realism audit reports observable public-trace proxies such as non-finite rates, zero runs, flatlines, spikes, and level shifts, while explicitly avoiding claims about unreleased operator ingestion-delay distributions.

Rebuttal-ready answer:

The strongest claims are explicitly restricted to resource-level operational telemetry. Alibaba and Salesforce/Borg are not described as service-level SLO traces, and GAIA/NetMan are used only for KPI stress diversity. The manuscript treats public-data realism as a boundary, not as a hidden assumption.

## Risk 2a: Stress operators are synthetic

Reviewer concern:

- Missing points, metric outage, delayed tail, burst, noise, and level shift may look manually chosen.
- A reviewer may ask whether these stresses reflect real AIOps monitoring failures.

Response:

- The stress taxonomy maps each operator to an operational source: scraper gaps, exporter failure, ingestion lag, workload spikes, release or migration shifts, and sensor or aggregation instability.
- The Stress Realism Audit reports public-trace proxies for zero runs, flatlines, spikes, and level shifts.
- The paper states that public traces are often cleaned and that the stress protocol brackets controlled severity rather than estimating operator-specific incident frequencies.
- The artifact includes natural proxy slice evaluation on Alibaba and Salesforce/Borg without synthetic corruption. This is reviewer-defense evidence for the concern that synthetic stress may be too artificial; it is mixed sanity evidence, not incident validation or service-level SLO validation.
- The artifact includes public fault-injection slice checks. RE1-OB is retained as caution evidence, while RE2-OB Online Boutique provides short-window public fault-injection sanity evidence: after splitting pre-injection normal data into train-normal and held-out-normal, 9 of 58 parsed cases change the best-MSE model from held-out-normal to fault windows, and 11 fault windows have different best-MSE and best-decision-cost models. The forecasters are trained per case with seed 42, not pooled across cases. The target is Istio P99 latency, so the proxy is labeled decision cost rather than resource-capacity cost for this probe.
- The RE2-OB numbers are now visible in a one-row main-paper sanity-check table, with the full fault-type breakdown kept in `outputs/public_fault_slice_v2/`.

Rebuttal-ready answer:

The stress protocol is synthetic by design but not arbitrary. AIOpsStressBench audits public traces for observable degradation proxies, adds public fault-injection slice sanity checks, and then injects controlled stress to evaluate model behavior under comparable mild/moderate/severe telemetry degradation. The paper does not claim to reproduce an unreleased failure distribution or validate incidents.

## Risk 2b: Simple imputation would solve the stress problem

Reviewer concern:

- Real AIOps pipelines often fill missing telemetry before forecasting.
- If forward-fill or mean-fill removes the degradation, the benchmark may overstate the difficulty.
- A reviewer may ask why no learned missing-aware preprocessing baseline is included.

Response:

- The imputation pipeline table uses a compact DLinear/PatchTST-lite check with no imputation and simple fill after stress injection.
- The rows are treated as preprocessing baselines, not as new forecasting backbones.
- The main paper reports that simple fill can reduce point-missing damage but does not remove channel-outage and stale-context model-selection risk.
- The artifact adds a bounded learned mask-aware imputer that reconstructs degraded input windows before the same DLinear and PatchTST-lite forecasters. It is not a SAITS/BRITS/PyPOTS benchmark, but it checks whether learned preprocessing changes the stressed pipeline choice.

Rebuttal-ready answer:

The benchmark evaluates preprocessing pipelines as well as forecasters. Simple imputation is included in the main table, and the artifact includes a bounded learned mask-aware imputation probe. These rows are not new forecasting backbones; they show that preprocessing itself is a pipeline choice and therefore should be evaluated under the same stress protocol.

## Risk 3: Capacity decision proxy is too simple

Reviewer concern:

- The simulator does not model scheduler constraints, queues, cold start, service dependencies, or true SLO.

Response:

- The manuscript consistently calls it a forecast-to-capacity decision proxy, not an autoscaling system.
- The default protocol is fixed: headroom `h=0.15`, under-provision cost `c_u=5.0`, over-provision cost `c_o=1.0`, demand floor `epsilon=0.05`, and target metric index `0`.
- The 5:1 cost ratio reflects the assumption that starvation/SLO risk is more costly than short-lived over-provisioning waste; artifact sensitivity recomputes 2:1, 5:1, and 10:1 ratios.
- It reports under-provision area, over-provision area, peak miss, severe-under rate, and total normalized cost.
- It includes reactive baselines to separate one-step reaction from proactive planning. Reactive HPA is a controller reference, not a forecasting model and not eligible for clean/stress winner selection.
- It includes horizon and headroom sensitivity on Alibaba and Salesforce/Borg.
- The main text now includes a compact capacity-sensitivity table with costs. For metric-channel outage, PatchTST-lite minimizes total decision cost on both multivariate sources at 2:1, 5:1, and 10:1 cost ratios and at horizon 96.

Rebuttal-ready answer:

The proxy is intentionally simple and auditable. It is used only for within-protocol comparisons of under-provisioning and over-provisioning consequences. The compact sensitivity table shows that the metric-outage decision-cost conclusion is not an artifact of one default cost ratio or horizon. The manuscript does not claim savings or autoscaling validity.

## Risk 4: Baseline fairness is weak

Reviewer concern:

- Official iTransformer LTSF bridge is not equivalent to the native per-entity NPZ protocol.
- Official baselines may be under-tuned.

Response:

- Official PatchTST is evaluated inside the native NPZ/stress pipeline.
- Native iTransformer wrapper is available for selected scenarios.
- Official TimeMixer is evaluated through THUML Time-Series-Library inside the native NPZ/stress pipeline on the two multivariate telemetry sources.
- Older LTSF-bridge iTransformer results are reference-only and should not drive the main ranking.
- LastValue is retained as a zero-parameter latency lower bound; learned-forecaster winner disagreements are summarized separately to avoid trivial latency claims.
- Main winner tables are restricted to the core learned comparable pool. LastValue, Chronos-Bolt, and LTSF-bridge iTransformer are excluded from core winner claims; all-available winners are kept only as artifact reference.
- The strongest conclusions are based on native-equivalent models, latency/memory metrics, and the decision proxy rather than a single leaderboard.
- The main paper summarizes baseline fairness in text to save space; the generated Baseline Fairness card records the detailed coverage and budget notes.

Rebuttal-ready answer:

Core ranking claims are based on native-equivalent protocols. Official wrappers, bridge runs, and zero-shot reference runs are separated in the artifact Baseline Fairness card. The bridge and Chronos-Bolt rows provide context but do not drive the central leaderboard or model-selection conclusions.

## Risk 4b: Foundation-model comparison is incomplete

Reviewer concern:

- The paper mentions Chronos, Moirai, and TimesFM, but only reports a small Chronos-Bolt reference.
- A reviewer may ask whether the conclusion changes with a larger or fine-tuned foundation model.

Response:

- The main claim is not that foundation models are weak. The claim is that deployment-stress evaluation must report latency, memory, and decision-proxy consequences in addition to clean MSE.
- Chronos-Bolt is included as a bounded zero-shot reference over Alibaba and Salesforce/Borg clean, missing-30, and missing-variable-30 windows.
- The reference shows feasibility and latency scale under the same stress framing, but the manuscript explicitly says this is not a full foundation-model benchmark.
- Chronos-Bolt is not eligible for the main winner table even when its bounded reference row has a favorable capacity proxy on a specific scenario.

Rebuttal-ready answer:

The paper does not claim that foundation models are weak or fully evaluated. Chronos-Bolt is included to provide a bounded zero-shot reference for feasibility, latency, and stress sensitivity. A complete Chronos/Moirai/TimesFM study is outside the current paper's main contribution.

## Risk 5: RACE-DLinear role is confusing

Reviewer concern:

- If RACE-DLinear is not a main contribution, why does it appear throughout the paper?

Response:

- RACE-DLinear is a lightweight robust baseline, not a new-model claim.
- Its ablation shows stress augmentation matters more than the wrapper itself.
- StressRoute v2 sometimes selects it because it sits between DLinear and PatchTST-lite in the latency-cost tradeoff.

Rebuttal-ready answer:

RACE-DLinear appears because the benchmark includes lightweight robust candidates, but the manuscript does not defend it as the main contribution. Its ablation supports the opposite conclusion: stress-aware evaluation matters more than the wrapper itself.

## Risk 6: StressRoute is too simple

Reviewer concern:

- A rule router or Logistic Regression router may be considered technically simple.

Response:

- For an Applied Track system paper, interpretability and auditability are advantages.
- StressRoute is not claimed as a complex learning algorithm. It operationalizes benchmark evidence into a decision-support policy.
- v1 provides transparent rules; v2 tests whether a lightweight learned policy can reduce latency-constrained capacity regret; oracle shows headroom.

Rebuttal-ready answer:

StressRoute is intentionally simple because the paper is an Applied benchmark/system evaluation. It operationalizes benchmark evidence into auditable model-selection rules, reports oracle gap, and is not claimed as a finished routing algorithm.

## Risk 7: Too many tables, weak narrative

Reviewer concern:

- The manuscript can read like an internal report.

Response:

- Keep only core evidence in the main paper:
  - pipeline figure;
  - benchmark card and stress taxonomy;
  - cross-source winner table;
  - stress robustness summary and multi-source severity figure;
  - deployment cost/forecast-to-capacity decision proxy and capacity-sensitivity compact table;
  - RE2-OB one-row sanity-check table;
  - multi-source deployment-stress figure;
  - compact case-study mechanism paragraph.
- Move detailed sensitivity or ablation tables to appendix/artifact when needed.

Rebuttal-ready answer:

The Results section is organized around model-selection questions rather than script order. Detailed sensitivity, ablation, policy, and case-overlay rows are moved to the artifact, while the main text preserves the evidence chain: pipeline, benchmark card, cross-source winner table, stress robustness summary, forecast-to-capacity decision proxy, capacity sensitivity, RE2-OB sanity check, and multi-source figure.

## Risk 8: Reproducibility claims are too broad

Reviewer concern:

- Large traces and multi-GPU experiments may be hard for reviewers to rerun.

Response:

- The artifact provides a minimal single-GPU reproduction path and separate commands for full sweeps.
- Each main table has a CSV source in `outputs/paper_tables`.
- Each main figure has a source CSV or generation script in `outputs/paper_figures`.
- Data acquisition attempts and caveats are documented.

Rebuttal-ready answer:

The artifact provides a minimum single-GPU path and separate commands for full sweeps. Generated tables and figures are linked to CSV/source files, and the manifest records datasets, stress definitions, metrics, and non-claims.
