# Submission Checklist

This checklist is for the final AIOpsStressBench ICDM Applied Track submission package.

## Paper

- [ ] `paper/main.pdf` is generated from `paper/main.tex`.
- [ ] `paper/main.pdf` is the canonical submission PDF; no alternate local PDFs are used for submission.
- [ ] PDF has at most 10 pages under the target IEEE format.
- [ ] `pdflatex -interaction=nonstopmode -halt-on-error main.tex` succeeds twice.
- [ ] LaTeX log has no `undefined`, `overfull`, `fatal`, or `LaTeX Error`.
- [ ] Author block is anonymous if the submission site requires anonymity.
- [ ] The paper does not contain temporary-version labels, local paths, host aliases, or user names.

## Claim Safety

- [ ] The central claim is consistently stated: clean LTSF evaluation can select different models under the controlled offline stress-to-decision protocol.
- [ ] Results open with the cross-source compact winner table, not an Alibaba-only result table.
- [ ] Baselines distinguish `PatchTST-lite` from `Official PatchTST`.
- [ ] The cross-source winner table includes the pool column and excludes LastValue, Chronos-Bolt, and LTSF-bridge references from learned-model selection.
- [ ] Figure 2(d) is objective-winner disagreement, not a routing-distribution panel.
- [ ] RACE-DLinear is described only as a lightweight robust baseline, not as a new architecture claim.
- [ ] StressRoute is described only as an auditable latency-constrained policy prototype, not an optimized router.
- [ ] Chronos-Bolt is described only as a bounded zero-shot reference, not a full foundation-model benchmark.
- [ ] Forecast-to-capacity evaluation is described only as a decision proxy, not an autoscaling system or savings claim.
- [ ] Capacity sensitivity compact rows include cost values and match `table_capacity_cost_ratio_sensitivity.csv` and `table_capacity_horizon_sensitivity.csv`.
- [ ] The main imputation table is compact and focuses on point missingness versus metric-channel outage.
- [ ] Structural findings are summarized in text; `structural_findings.tex` is not referenced from the main paper.
- [ ] Alibaba and Salesforce/Borg are described as resource-level operational telemetry, not service-level SLO traces.
- [ ] LastValue, Chronos-Bolt, and LTSF-bridge iTransformer are excluded from the core learned winner pool.
- [ ] GAIA and NetMan are used only for KPI stress diversity and case studies, not resource-capacity conclusions.
- [ ] Main-paper five-seed claims trace to `outputs/five_seed`, not old three-seed summaries.
- [ ] RE2-OB numbers in the paper match `outputs/public_fault_slice_v2/fault_slice_v2_decision.md`.

## Artifact

- [ ] `ANON_ARTIFACT.md` is the artifact entry point.
- [ ] `benchmark_manifest.yaml` passes `python scripts/check_benchmark_manifest.py --manifest benchmark_manifest.yaml --root .`.
- [ ] Main paper tables can be traced to `outputs/paper_tables/`.
- [ ] Main paper figures can be traced to `outputs/paper_figures/`.
- [ ] `paper/tables/cross_source_winners.tex`, `capacity_sensitivity_compact.tex`, and `public_fault_mini.tex` are generated from existing outputs only.
- [ ] Five-seed tables can be traced to `outputs/five_seed/`.
- [ ] RE2-OB public fault-injection sanity check can be traced to `outputs/public_fault_slice_v2/`.
- [ ] Defensive probes are present under `outputs/strong_probe/defensive_tables/`.
- [ ] Chronos-Bolt A6000 reference is present under `outputs/strong_probe/foundation_reference_a6000/`.
- [ ] Artifact package excludes `.venv/`, `.git/`, raw third-party large data, SSH files, internal host logs, and checkpoint files.

## Anonymity

- [ ] Search the paper and anonymous artifact for personal names, host aliases, IP addresses, local absolute paths, and home-directory paths.
- [ ] Public artifact docs use relative paths or `<repo_root>` style paths.
- [ ] Internal handoff package is kept separate from anonymous submission artifact.

## Dry Run

- [ ] Unzip the anonymous artifact into a temporary directory.
- [ ] Run the manifest check inside the unzipped directory.
- [ ] Confirm `paper/main.pdf`, `paper/main.tex`, `ANON_ARTIFACT.md`, `FINAL_CLAIM_AUDIT.md`, and `benchmark_manifest.yaml` are present.
- [ ] Confirm the unzipped package has no `.venv/` or `.git/`.
