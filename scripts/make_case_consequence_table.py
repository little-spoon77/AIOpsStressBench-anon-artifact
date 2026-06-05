#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    fig = root / "outputs" / "paper_figures"
    out_csv = root / "outputs" / "paper_tables" / "table_case_consequence.csv"
    out_tex = root / "paper" / "tables" / "case_consequence.tex"

    cases = [
        ("Telemetry outage", "netman_missing_variables_30_metrics.csv", "min_local_capacity"),
        ("Delayed telemetry", "salesforce_borg_delayed_12_metrics.csv", "min_local_capacity"),
        ("Point missingness", "netman_missing_30_metrics.csv", "min_local_capacity"),
        ("Metric outage", "alibaba2018_missing_variables_30_metrics.csv", "min_global_capacity"),
    ]
    rows = []
    for label, filename, mode in cases:
        df = pd.read_csv(fig / filename)
        if mode == "min_global_capacity":
            row = df.sort_values(["global_capacity_cost", "latency_p95_ms"]).iloc[0]
            cost = float(row["global_capacity_cost"])
        else:
            row = df.sort_values(["local_capacity_cost", "latency_p95_ms"]).iloc[0]
            cost = float(row["local_capacity_cost"])
        rows.append(
            {
                "case": label,
                "selected_model": row["model"],
                "capacity_cost": cost,
                "under_rate": float(row["local_under_rate"]),
                "p95_latency_ms": float(row["latency_p95_ms"]),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Deployment consequences in selected case-study windows. Cost is local capacity proxy except for the Alibaba metric-outage row, where global capacity proxy identifies the deployment-preferred model.}",
        r"\label{tab:case-consequence}",
        r"\scriptsize",
        r"\begin{tabular}{l l r r r}",
        r"\toprule",
        r"Case & Selected & Cost & Under & P95 ms \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['case']} & {row['selected_model']} & "
            f"{row['capacity_cost']:.3f} & {row['under_rate']:.1f} & "
            f"{row['p95_latency_ms']:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
