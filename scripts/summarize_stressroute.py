from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def pct_change(value: float, baseline: float) -> float:
    if pd.isna(value) or pd.isna(baseline) or baseline == 0:
        return float("nan")
    return (value - baseline) / baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize StressRoute results against fixed baselines and oracle.")
    parser.add_argument("--input", default="outputs/stressroute_v1_alibaba_linear.csv")
    parser.add_argument("--selection", default="outputs/stressroute_v1_alibaba_linear_selection.csv")
    parser.add_argument("--output", default="outputs/stressroute_v1_alibaba_linear_report.csv")
    parser.add_argument("--selection-output", default="outputs/stressroute_v1_alibaba_linear_selection_report.csv")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    rows = []
    for keys, group in frame.groupby(["source", "dataset", "stress"], sort=True):
        source, dataset, stress = keys
        fixed = group[group["policy"] == "fixed"].copy()
        oracle = group[group["policy"] == "oracle_mse"].copy()
        fixed_by_model = {row["model"]: row for _, row in fixed.iterrows()}
        oracle_row = oracle.iloc[0] if not oracle.empty else None
        for _, row in group[group["policy"].astype(str).str.startswith("stressroute_v1")].iterrows():
            dlinear = fixed_by_model.get("dlinear")
            route = {
                "source": source,
                "dataset": dataset,
                "stress": stress,
                "policy": row["policy"],
                "latency_budget_ms": row.get("latency_budget_ms"),
                "selected_model": row["model"],
                "route_reason": row.get("route_reason"),
                "mse": row["mse"],
                "mae": row["mae"],
                "capacity_cost": row["capacity_cost"],
                "latency_p95_ms": row["latency_p95_ms"],
            }
            if dlinear is not None:
                route["mse_vs_dlinear"] = pct_change(row["mse"], dlinear["mse"])
                route["capacity_cost_vs_dlinear"] = pct_change(row["capacity_cost"], dlinear["capacity_cost"])
                route["latency_vs_dlinear"] = pct_change(row["latency_p95_ms"], dlinear["latency_p95_ms"])
            if oracle_row is not None:
                route["mse_oracle_gap"] = pct_change(row["mse"], oracle_row["mse"])
                route["capacity_oracle_gap"] = pct_change(row["capacity_cost"], oracle_row["capacity_cost"])
            rows.append(route)

    report = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)

    selection_path = Path(args.selection)
    if selection_path.exists():
        selection = pd.read_csv(selection_path)
        counts = (
            selection.groupby(["source", "dataset", "stress", "policy", "selected_model"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        counts["share"] = counts["count"] / counts.groupby(["source", "dataset", "stress", "policy"])["count"].transform("sum")
        selection_output = Path(args.selection_output)
        selection_output.parent.mkdir(parents=True, exist_ok=True)
        counts.to_csv(selection_output, index=False)
        print(f"Saved {selection_output}")

    print(report.to_string(index=False))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
