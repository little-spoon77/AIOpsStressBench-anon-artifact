#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path("outputs/foundation_reference")
    files = sorted(root.glob("chronos_bolt_*_*_a6000.csv"))
    files = [p for p in files if not p.name.startswith("smoke_")]
    if not files:
        raise SystemExit("No A6000 Chronos CSV files found.")

    frames = []
    for path in files:
        df = pd.read_csv(path)
        df["file"] = path.name
        frames.append(df)
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(root / "foundation_reference_summary.csv", index=False)

    clean = summary[summary["stress"] == "clean"][
        ["dataset", "model_id", "mse", "capacity_cost", "latency_p95_ms", "max_memory_mb"]
    ].rename(
        columns={
            "mse": "clean_mse",
            "capacity_cost": "clean_capacity_cost",
            "latency_p95_ms": "clean_p95_ms",
            "max_memory_mb": "memory_mb",
        }
    )
    merged = summary.merge(clean, on=["dataset", "model_id"], how="left")
    merged["mse_degradation_vs_clean"] = merged["mse"] / merged["clean_mse"].clip(lower=1e-12)
    merged["capacity_degradation_vs_clean"] = (
        merged["capacity_cost"] / merged["clean_capacity_cost"].clip(lower=1e-12)
    )

    compact_cols = [
        "dataset",
        "stress",
        "model_id",
        "eval_windows",
        "mse",
        "mae",
        "capacity_cost",
        "mse_degradation_vs_clean",
        "capacity_degradation_vs_clean",
        "latency_p95_ms",
        "max_memory_mb",
        "zero_shot",
    ]
    compact = merged[compact_cols].copy()
    compact.to_csv(root / "foundation_reference_compact_table.csv", index=False)

    failures = root / "foundation_reference_failures.md"
    failures.write_text(
        "\n".join(
            [
                "# Foundation Reference Failures / Scope",
                "",
                "- Chronos-Bolt tiny/small/base completed on A6000 GPU1 with local-uploaded HuggingFace cache.",
                "- The execution machine had no outbound network access; model caches were staged before evaluation.",
                "- TimesFM and Moirai were not run in this A6000 pass. They remain optional reference-only extensions, not part of the core benchmark claim.",
                "- No fine-tuning was performed; all rows are bounded zero-shot reference results.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    lines = [
        "# Foundation Reference Decision",
        "",
        "## Verdict",
        "",
        "Use Chronos-Bolt as a bounded reference table or artifact evidence, not as a core winner table model.",
        "",
        "## Evidence Summary",
        "",
    ]
    for dataset in sorted(summary["dataset"].unique()):
        lines.append(f"### {dataset}")
        part = merged[merged["dataset"] == dataset]
        for model_id in sorted(part["model_id"].unique()):
            clean_row = part[(part["model_id"] == model_id) & (part["stress"] == "clean")].iloc[0]
            worst = part[part["model_id"] == model_id].sort_values("mse", ascending=False).iloc[0]
            lines.append(
                "- `{}`: clean MSE {:.6g}, worst stress `{}` MSE {:.6g}, "
                "P95 latency {:.2f} ms, peak memory {:.0f} MB.".format(
                    model_id,
                    clean_row["mse"],
                    worst["stress"],
                    worst["mse"],
                    clean_row["latency_p95_ms"],
                    clean_row["max_memory_mb"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Paper Use",
            "",
            "- Supports the claim that foundation models are not a free robustness fix under strict online latency.",
            "- Suitable for a small reference table or appendix/artifact, because the setup is zero-shot and not fine-tuned.",
            "- Do not include Chronos-Bolt in the core comparable winner pool.",
            "",
        ]
    )
    (root / "foundation_reference_decision.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {root / 'foundation_reference_summary.csv'}")
    print(f"Saved {root / 'foundation_reference_compact_table.csv'}")
    print(f"Saved {root / 'foundation_reference_decision.md'}")


if __name__ == "__main__":
    main()
