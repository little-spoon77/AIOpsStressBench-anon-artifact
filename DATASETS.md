# AIOpsStressBench Dataset Plan

This benchmark uses public operational time-series data only. Private or non-reproducible data must not be used for main claims.

## Unified Format

Every processed dataset should be exported as:

```text
series: float32 [entities, time, metrics]
metric_names: string [metrics]
entity_ids: string [entities]
metadata: optional object with dataset name, source URL, license, role, and caveat
```

Required derived files:

```text
data/<dataset>.npz
outputs/dataset_audit_<dataset>.csv
```

Audit fields:

- entities
- time steps
- metrics
- missing rate
- zero rate
- mean / std / min / max
- source URL
- role
- caveat

## Main Data Sources

| Dataset | Status | Role | Caveat |
|---|---|---|---|
| Alibaba Cluster Trace 2018 | available | machine-level multivariate resource telemetry | not service-level request/SLO data |
| NetMan KPI | available | KPI telemetry and metric-outage cases | KPI semantics are not resource capacity |
| GAIA | available | behavior diversity and KPI stress categories | mostly single-target KPI series |
| Salesforce CloudOps TSF / Borg 2011 | expanded candidate | CloudOps workload/resource traces | 256-entity subset; `dynamic_4` is all zero and should not be interpreted as an active metric |
| Azure VM traces | target | public VM workload traces | may require custom aggregation into multivariate windows |
| Google/Borg traces | target/reference | large production workload trace | large scale; use only reproducible subset |

## Salesforce CloudOps TSF Candidate

The official HuggingFace endpoint was unreliable from multiple execution
environments, but the mirror endpoint exposes the same repository metadata and raw
zip files:

```bash
curl -L -I https://hf-mirror.com/datasets/Salesforce/cloudops_tsf/resolve/main/borg_cluster_data_2011/train_test.zip
```

Current successful raw file:

```text
data/raw/salesforce_cloudops/borg_cluster_data_2011_train_test.zip
```

Converted tensors:

```text
data/salesforce_borg_64x1200.npz
outputs/dataset_audit_salesforce_borg_64x1200.csv
data/salesforce_borg_256x2048.npz
outputs/dataset_audit_salesforce_borg_256x2048.csv
```

Expanded audit summary:

- entities: 256
- time steps: 2048
- metrics: 7
- finite rate after transparent interpolation: 1.0
- original raw non-finite rate is stored in NPZ metadata
- source license from HuggingFace repo metadata: CC-BY-4.0
- caveat: `dynamic_4` is all zero in the current subset and must not be interpreted as a meaningful operational metric

Stress and routing results:

```text
outputs/salesforce_borg_64x1200_stress_summary.csv
outputs/salesforce_borg_256x2048_stress_summary.csv
outputs/stressroute_v1_salesforce_borg_256x2048.csv
outputs/stressroute_v1_salesforce_borg_256x2048_report.csv
outputs/official_patchtst_salesforce_borg_256x2048_summary.csv
outputs/official_itransformer_salesforce_borg_256x2048_summary.csv
outputs/severity_curve_salesforce_borg_256x2048_summary.csv
outputs/multiseed_salesforce_borg_256x2048_summary.csv
```

Decision: Salesforce/Borg is now the second multivariate operational telemetry
candidate alongside Alibaba. It now has native stress, StressRoute, official
baseline, stress-severity, and three-seed stability evidence. For the ICDM 2027
version, it can support the multi-source operational telemetry story with the
explicit caveat that `dynamic_4` is all zero in the current subset.

## Azure VM CPU Candidate

The script `scripts/prepare_azure_vm_cpu.py` converts Azure VM CPU readings into the unified NPZ format when a complete or explicitly partial gzip is available. It exports three CPU metrics:

```text
min_cpu, max_cpu, avg_cpu
```

Current diagnostic on the complete first CPU-reading shard:

- `schema.csv` is available.
- `vm_cpu_readings-file-1-of-195.full.csv.gz` downloaded successfully.
- The file size matches Azure Blob `content-length` and passes gzip validation.
- The shard has 10,000,000 rows, 241,490 VMs, and only 45 timestamps.
- No entity satisfies a 128-step or 1,000-step forecasting threshold in this shard.

Decision: Azure remains a high-value 2027 target, but a single CPU shard is not
yet a main benchmark dataset. It can only become a main dataset after multiple
shards are stitched or a reproducible aggregation strategy yields enough time
steps per entity.

Current acquisition tools:

- `scripts/download_azure_public_dataset.py` downloads selected Azure files with content-length and gzip checks.
- `scripts/inspect_azure_vm_cpu.py` checks whether a CPU-reading file contains enough timestamps per VM for forecasting.
- `scripts/prepare_azure_vm_cpu.py` converts a complete or explicitly partial CPU-reading file into NPZ format.

Do not use Azure in main paper claims until a complete file passes gzip validation and the inspection step reports enough entities with sufficient time length.

## Acceptance Criteria for ICDM 2027

Minimum final benchmark:

- At least 4 public datasets can be audited with one command.
- At least 2 datasets are multivariate workload/resource traces.
- At least 1 dataset is not derived from Alibaba.
- Each dataset has an explicit role and caveat in the paper.
- No main result depends on data that cannot be downloaded or regenerated.

## Immediate Data Tasks

1. Rewrite the paper dataset and evaluation sections around Alibaba plus Salesforce/Borg as the two multivariate operational telemetry sources.
2. Consider expanding Salesforce/Borg to 512 entities if runtime remains acceptable.
3. Keep Alibaba as the stable machine-resource baseline.
4. Keep NetMan and GAIA for stress diversity and failure-mode case studies.

## Non-Claims

- Alibaba is not service-level request telemetry.
- GAIA and NetMan do not validate production capacity savings.
- The benchmark reports decision proxies, not measured production autoscaling savings.
