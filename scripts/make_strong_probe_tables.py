from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _read_existing(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No input CSV files exist.")
    return pd.concat(frames, ignore_index=True)


def make_stressroute_regret(output_dir: Path) -> None:
    df = _read_existing(
        [
            Path("outputs/stressroute_v2_alibaba_mixed.csv"),
            Path("outputs/stressroute_v2_salesforce_borg_mixed.csv"),
        ]
    )
    df = df[df["objective"].isin(["capacity", "mse"])].copy()
    df["latency_budget_ms"] = df["latency_budget_ms"].astype(str)
    df["capacity_regret"] = df["capacity_oracle_gap"]
    df["mse_regret"] = df["mse_oracle_gap"]

    policy_order = ["fixed_dlinear", "fixed_patchtst", "stressroute_v1", "stressroute_v2", "oracle"]
    summary = (
        df.groupby(["dataset", "objective", "latency_budget_ms", "policy"], dropna=False)
        .agg(
            scenarios=("stress", "nunique"),
            mean_capacity_cost=("capacity_cost", "mean"),
            mean_mse=("mse", "mean"),
            mean_latency_p95_ms=("latency_p95_ms", "mean"),
            feasible_rate=("budget_feasible", "mean"),
            mean_capacity_regret=("capacity_regret", "mean"),
            mean_mse_regret=("mse_regret", "mean"),
        )
        .reset_index()
    )
    summary["policy"] = pd.Categorical(summary["policy"], policy_order, ordered=True)
    summary = summary.sort_values(["dataset", "objective", "latency_budget_ms", "policy"])

    selection = (
        df[df["policy"].isin(["stressroute_v1", "stressroute_v2"])]
        .groupby(["dataset", "objective", "latency_budget_ms", "policy", "selected_model"], dropna=False)
        .size()
        .reset_index(name="selected_count")
        .sort_values(["dataset", "objective", "latency_budget_ms", "policy", "selected_count"], ascending=[True, True, True, True, False])
    )

    compact_rows = []
    for (dataset, objective, budget), group in summary.groupby(["dataset", "objective", "latency_budget_ms"], observed=False):
        route = group[(group["policy"].isin(["stressroute_v1", "stressroute_v2"])) & (group["feasible_rate"] >= 1.0)].copy()
        fixed = group[(group["policy"].isin(["fixed_dlinear", "fixed_patchtst"])) & (group["feasible_rate"] >= 1.0)].copy()
        oracle = group[group["policy"] == "oracle"].copy()
        if route.empty or fixed.empty:
            continue
        best_route = route.sort_values(["mean_capacity_regret", "mean_mse_regret"]).iloc[0]
        best_fixed = fixed.sort_values(["mean_capacity_regret", "mean_mse_regret"]).iloc[0]
        oracle_cost = float(oracle["mean_capacity_cost"].iloc[0]) if not oracle.empty else np.nan
        compact_rows.append(
            {
                "dataset": dataset,
                "objective": objective,
                "latency_budget_ms": budget,
                "best_route_policy": best_route["policy"],
                "best_route_capacity_regret": best_route["mean_capacity_regret"],
                "best_fixed_policy": best_fixed["policy"],
                "best_fixed_capacity_regret": best_fixed["mean_capacity_regret"],
                "route_minus_fixed_regret": best_route["mean_capacity_regret"] - best_fixed["mean_capacity_regret"],
                "oracle_capacity_cost": oracle_cost,
                "route_p95_ms": best_route["mean_latency_p95_ms"],
                "route_feasible_rate": best_route["feasible_rate"],
            }
        )
    compact = pd.DataFrame(compact_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "stressroute_regret_long.csv", index=False)
    summary.to_csv(output_dir / "stressroute_regret_summary.csv", index=False)
    selection.to_csv(output_dir / "stressroute_selection_distribution.csv", index=False)
    compact.to_csv(output_dir / "stressroute_regret_compact.csv", index=False)

    report = [
        "# StressRoute regret probe",
        "",
        "This probe reuses existing StressRoute v1/v2 outputs and computes oracle-gap/regret summaries.",
        "It is a defensive analysis only; it does not change the core paper tables.",
        "",
        "## Files",
        "",
        "- `stressroute_regret_long.csv`",
        "- `stressroute_regret_summary.csv`",
        "- `stressroute_selection_distribution.csv`",
        "- `stressroute_regret_compact.csv`",
        "",
        "## Compact view",
        "",
        compact.to_markdown(index=False) if not compact.empty else "No compact rows generated.",
        "",
    ]
    (output_dir / "stressroute_regret_decision.md").write_text("\n".join(report), encoding="utf-8")


def make_multiseed_compact(output_dir: Path) -> None:
    df = _read_existing(
        [
            Path("outputs/multiseed_summary.csv"),
            Path("outputs/multiseed_salesforce_borg_256x2048_summary.csv"),
        ]
    )
    core_models = ["dlinear", "race_dlinear", "patchtst"]
    stresses = ["clean", "missing_30", "missing_variables_30", "delayed_12", "level_shift"]
    df = df[df["model"].isin(core_models) & df["stress"].isin(stresses)].copy()

    stats = (
        df.groupby(["dataset", "stress", "model"])
        .agg(
            seeds=("seed", "nunique"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            capacity_cost_mean=("capacity_cost", "mean"),
            capacity_cost_std=("capacity_cost", "std"),
            latency_p95_mean=("latency_p95_ms", "mean"),
            latency_p95_std=("latency_p95_ms", "std"),
        )
        .reset_index()
    )
    winners = []
    for (dataset, stress), group in stats.groupby(["dataset", "stress"]):
        winners.append(
            {
                "dataset": dataset,
                "stress": stress,
                "best_mse_model": group.sort_values(["mse_mean", "model"]).iloc[0]["model"],
                "best_capacity_model": group.sort_values(["capacity_cost_mean", "model"]).iloc[0]["model"],
                "lowest_latency_model": group.sort_values(["latency_p95_mean", "model"]).iloc[0]["model"],
                "models": len(group),
                "min_seeds": int(group["seeds"].min()),
            }
        )
    winners_frame = pd.DataFrame(winners)
    winners_frame["objective_disagreement"] = (
        (winners_frame["best_mse_model"] != winners_frame["best_capacity_model"])
        | (winners_frame["best_mse_model"] != winners_frame["lowest_latency_model"])
    )

    compact = (
        winners_frame.groupby("dataset")
        .agg(
            scenarios=("stress", "nunique"),
            scenarios_with_objective_disagreement=("objective_disagreement", "sum"),
            min_seeds=("min_seeds", "min"),
        )
        .reset_index()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_dir / "multiseed_core_stats.csv", index=False)
    winners_frame.to_csv(output_dir / "multiseed_core_winners.csv", index=False)
    compact.to_csv(output_dir / "multiseed_core_compact.csv", index=False)
    report = [
        "# Multi-seed stability probe",
        "",
        "This probe summarizes existing 3-seed runs for the native lightweight core models.",
        "Official-model expanded seeds are tracked separately because they require new GPU training.",
        "",
        compact.to_markdown(index=False) if not compact.empty else "No compact rows generated.",
        "",
    ]
    (output_dir / "multiseed_stability_decision.md").write_text("\n".join(report), encoding="utf-8")


def make_expanded_multiseed_compact(output_dir: Path) -> None:
    expanded_dir = Path("outputs/strong_probe/expanded_multiseed")
    seed42_sources = [
        Path("outputs/official_patchtst_native_summary.csv"),
        Path("outputs/official_patchtst_salesforce_borg_256x2048_summary.csv"),
        Path("outputs/official_itransformer_native_summary.csv"),
        Path("outputs/official_itransformer_salesforce_borg_256x2048_summary.csv"),
        Path("outputs/official_timemixer_native_summary.csv"),
        Path("outputs/official_timemixer_salesforce_borg_256x2048_summary.csv"),
    ]
    frames = []
    for path in seed42_sources:
        if path.exists():
            frame = pd.read_csv(path)
            frame["seed"] = 42
            frames.append(frame)
    if expanded_dir.exists():
        for path in sorted(expanded_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            stem = path.stem
            seed = None
            if "_seed_" in stem:
                try:
                    seed = int(stem.rsplit("_seed_", 1)[1])
                except ValueError:
                    seed = None
            if seed is not None:
                frame["seed"] = seed
            frames.append(frame)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    stresses = ["clean", "missing_30", "missing_variables_30", "delayed_12", "level_shift"]
    models = ["official_patchtst", "official_itransformer_native", "official_timemixer"]
    df = df[df["stress"].isin(stresses) & df["model"].isin(models)].copy()
    if df.empty:
        return
    stats = (
        df.groupby(["dataset", "stress", "model"])
        .agg(
            seeds=("seed", "nunique"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            capacity_cost_mean=("capacity_cost", "mean"),
            capacity_cost_std=("capacity_cost", "std"),
            latency_p95_mean=("latency_p95_ms", "mean"),
            latency_p95_std=("latency_p95_ms", "std"),
        )
        .reset_index()
        .sort_values(["dataset", "stress", "model"])
    )
    winners = []
    complete = stats[stats["seeds"] >= 3].copy()
    for (dataset, stress), group in complete.groupby(["dataset", "stress"]):
        if len(group) < 2:
            continue
        winners.append(
            {
                "dataset": dataset,
                "stress": stress,
                "models_with_3_seeds": len(group),
                "best_mse_model": group.sort_values(["mse_mean", "model"]).iloc[0]["model"],
                "best_capacity_model": group.sort_values(["capacity_cost_mean", "model"]).iloc[0]["model"],
                "lowest_latency_model": group.sort_values(["latency_p95_mean", "model"]).iloc[0]["model"],
            }
        )
    winners_frame = pd.DataFrame(winners)
    if not winners_frame.empty:
        winners_frame["objective_disagreement"] = (
            (winners_frame["best_mse_model"] != winners_frame["best_capacity_model"])
            | (winners_frame["best_mse_model"] != winners_frame["lowest_latency_model"])
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(output_dir / "expanded_multiseed_official_stats.csv", index=False)
    winners_frame.to_csv(output_dir / "expanded_multiseed_official_winners.csv", index=False)
    report = [
        "# Expanded official-model multi-seed probe",
        "",
        "This table combines existing seed-42 official runs with any new seed-2025/2026 CSVs under `outputs/strong_probe/expanded_multiseed`.",
        "Incomplete rows are kept in the CSV but excluded from the winner summary.",
        "",
        "## Complete winner rows",
        "",
        winners_frame.to_markdown(index=False) if not winners_frame.empty else "No complete 3-seed official rows yet.",
        "",
    ]
    (output_dir / "expanded_multiseed_official_decision.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate defensive strong-probe tables without changing paper outputs.")
    parser.add_argument("--output-dir", default="outputs/strong_probe/defensive_tables")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    make_stressroute_regret(output_dir)
    make_multiseed_compact(output_dir)
    make_expanded_multiseed_compact(output_dir)
    print(f"Saved defensive probe tables to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
