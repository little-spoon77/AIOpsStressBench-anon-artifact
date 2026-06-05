from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def add_relative_columns(frame: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    enriched = []
    for scenario, group in frame.groupby("scenario", sort=False):
        group = group.copy()
        ref = group[group["model"] == reference_model]
        if ref.empty:
            group["mse_vs_ref_pct"] = pd.NA
            group["latency_p95_vs_ref_pct"] = pd.NA
            group["capacity_cost_vs_ref_pct"] = pd.NA
        else:
            ref_row = ref.iloc[0]
            group["mse_vs_ref_pct"] = (group["mse"] / ref_row["mse"] - 1.0) * 100.0
            group["latency_p95_vs_ref_pct"] = (group["latency_p95_ms"] / ref_row["latency_p95_ms"] - 1.0) * 100.0
            group["capacity_cost_vs_ref_pct"] = (group["capacity_cost"] / ref_row["capacity_cost"] - 1.0) * 100.0
        enriched.append(group)
    return pd.concat(enriched, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics.csv files from stress-suite outputs.")
    parser.add_argument("--root", default="outputs/stress_suite", help="Directory containing scenario subdirectories.")
    parser.add_argument("--output", default=None, help="Optional output CSV path.")
    parser.add_argument("--reference-model", default="dlinear", help="Model used for relative deltas.")
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for metrics_path in sorted(root.glob("*/metrics.csv")):
        scenario = metrics_path.parent.name
        frame = pd.read_csv(metrics_path)
        frame.insert(0, "scenario", scenario)
        rows.append(frame)
    if not rows:
        raise SystemExit(f"No metrics.csv files found under {root}")

    result = pd.concat(rows, ignore_index=True)
    result = add_relative_columns(result, args.reference_model)
    output = Path(args.output) if args.output else root / "summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.sort_values(["scenario", "mse"]).to_string(index=False))
    print(f"\nSaved summary to: {output.resolve()}")


if __name__ == "__main__":
    main()
