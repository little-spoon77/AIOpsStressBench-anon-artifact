# Paper Source

This directory contains the current IEEE-style LaTeX source.

## Build

From `hsc2/paper`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The current local build produces:

- `main.pdf`

Known build status:

- The source compiles with MiKTeX `pdflatex`.
- Remaining messages are ordinary underfull box warnings.
- Tables are generated from `outputs/paper_tables` by:

```bash
python ../scripts/make_latex_tables.py \
  --table-dir ../outputs/paper_tables \
  --output-dir tables
```

The table snippets currently include dataset audit, Alibaba clean accuracy,
Alibaba deployment-stress robustness, Alibaba deployment cost, Alibaba/NetMan
capacity-risk proxy, forecast-to-capacity simulator results, scenario winners,
Alibaba and Salesforce/Borg severity-curve summaries, Alibaba and
Salesforce/Borg multi-seed stability, StressRoute v1 policy results, GAIA
category winners, and the RACE-DLinear ablation.

## Current Scope

This is a first LaTeX skeleton, not a submission-ready paper. It already
contains the main story, core tables, accuracy-latency figure, and case-study
figure placeholders.

## Experiment Snapshot

Current paper results use GAIA metric forecast, NetMan KPI, Alibaba Cluster
Trace 2018 machine usage, and Salesforce CloudOps/Borg 2011. Alibaba and
Salesforce/Borg are the two multivariate operational telemetry sources; GAIA
and NetMan provide KPI stress diversity and failure-mode case studies. Native
baselines are LastValue, DLinear, RACE-DLinear, and lightweight PatchTST-lite.
Official baselines are official PatchTST and native iTransformer through the
native NPZ/stress pipeline for selected Alibaba and Salesforce/Borg scenarios.

New sprint outputs, when present:

- `outputs/severity_curve_summary.csv`
- `outputs/severity_curve_salesforce_borg_256x2048_summary.csv`
- `outputs/official_patchtst_severity_summary.csv`
- `outputs/official_itransformer_native_summary.csv`
- `outputs/official_patchtst_salesforce_borg_256x2048_summary.csv`
- `outputs/official_itransformer_salesforce_borg_256x2048_summary.csv`
- `outputs/multiseed_summary.csv`
- `outputs/multiseed_salesforce_borg_256x2048_summary.csv`
- `outputs/capacity_simulator_sensitivity_summary.csv`
- `outputs/stressroute_v1_salesforce_borg_256x2048_report.csv`

Alibaba official PatchTST stress coverage now includes clean, missing_10,
missing_30, missing_50, missing_variables_30, delayed_12, noisy, burst, and
level_shift.

Keep these caveats in the paper text: RACE-DLinear is not the main
contribution; the capacity simulator is a deployment proxy, not a production
autoscaling simulator; Alibaba is machine-level resource telemetry, not
service-level request telemetry; the current Salesforce/Borg subset has one
all-zero dynamic metric; old LTSF-bridge iTransformer results are reference
only and should not be used as central ranking claims.




