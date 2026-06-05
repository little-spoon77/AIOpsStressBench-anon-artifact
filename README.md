# AIOpsStressBench Anonymous Review Artifact

This repository is the review artifact for `AIOpsStressBench: An Offline Stress-to-Decision Benchmark for Operational Time-Series Forecasting`.

Start with `ANON_ARTIFACT.md` for the reviewer-facing guide. The package includes processed public NPZ tensors, dataset/preparation notes, stress operators, model wrappers, generated paper tables and figures, five-seed summaries, RE2-OB sanity-check outputs, and scripts for table/figure generation.

Quick check:

```bash
python scripts/check_benchmark_manifest.py --manifest benchmark_manifest.yaml --root .
```

The canonical paper source is `paper/main.tex`; the canonical generated PDF is `paper/main.pdf`.
