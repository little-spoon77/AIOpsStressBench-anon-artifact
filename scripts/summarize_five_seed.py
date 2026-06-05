from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CORE_MODELS = ["dlinear", "race_dlinear", "patchtst"]
OFFICIAL_MODELS = ["official_patchtst", "official_itransformer_native", "official_timemixer"]
SCENARIOS = ["clean", "missing_30", "missing_variables_30", "delayed_12", "level_shift"]


def to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def read_existing(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def normalize_dataset(value: str) -> str:
    mapping = {
        "alibaba2018": "alibaba2018",
        "alibaba2018_machine_usage": "alibaba2018",
        "salesforce_borg": "salesforce_borg",
        "salesforce_borg_256x2048": "salesforce_borg",
    }
    return mapping.get(str(value), str(value))


def load_core(summary_paths: list[Path]) -> pd.DataFrame:
    frame = read_existing(summary_paths)
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].map(normalize_dataset)
    frame["pool"] = "core_lightweight"
    return frame


def load_official(seed42_paths: list[Path], expanded_dir: Path) -> pd.DataFrame:
    frames = []
    for path in seed42_paths:
        if path.exists():
            frame = pd.read_csv(path)
            frame["seed"] = 42
            frames.append(frame)
    if expanded_dir.exists():
        for path in sorted(expanded_dir.glob("*.csv")):
            frame = pd.read_csv(path)
            if "seed" not in frame.columns:
                seed = None
                stem = path.stem
                if "_seed_" in stem:
                    try:
                        seed = int(stem.rsplit("_seed_", 1)[1])
                    except ValueError:
                        seed = None
                if seed is not None:
                    frame["seed"] = seed
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    frame["dataset"] = frame["dataset"].map(normalize_dataset)
    frame["pool"] = "official_probe"
    return frame


def summarize(frame: pd.DataFrame, output_dir: Path, expected_seeds: set[int]) -> None:
    needed = ["dataset", "stress", "model", "seed", "mse", "mae", "capacity_cost", "latency_p95_ms"]
    missing_cols = [col for col in needed if col not in frame.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    frame = frame[frame["stress"].isin(SCENARIOS)].copy()
    frame = frame[frame["model"].isin(CORE_MODELS + OFFICIAL_MODELS)].copy()
    frame["dataset"] = frame["dataset"].map(normalize_dataset)
    frame = frame.drop_duplicates(subset=["pool", "dataset", "stress", "model", "seed"], keep="last")

    stats = (
        frame.groupby(["pool", "dataset", "stress", "model"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            capacity_cost_mean=("capacity_cost", "mean"),
            capacity_cost_std=("capacity_cost", "std"),
            latency_p95_mean=("latency_p95_ms", "mean"),
            latency_p95_std=("latency_p95_ms", "std"),
        )
        .reset_index()
        .sort_values(["pool", "dataset", "stress", "model"])
    )
    stats["complete_5_seed"] = stats["seeds"] >= len(expected_seeds)

    winners = []
    for (pool, dataset, stress), group in stats.groupby(["pool", "dataset", "stress"], dropna=False):
        complete = group[group["complete_5_seed"]].copy()
        candidate = complete if not complete.empty else group.copy()
        if candidate.empty:
            continue
        winners.append(
            {
                "pool": pool,
                "dataset": dataset,
                "stress": stress,
                "models": int(len(candidate)),
                "min_seeds": int(candidate["seeds"].min()),
                "complete_5_seed_models": int(complete["model"].nunique()),
                "best_mse_model": candidate.sort_values(["mse_mean", "model"]).iloc[0]["model"],
                "best_capacity_model": candidate.sort_values(["capacity_cost_mean", "model"]).iloc[0]["model"],
                "lowest_latency_model": candidate.sort_values(["latency_p95_mean", "model"]).iloc[0]["model"],
            }
        )
    winners_frame = pd.DataFrame(winners)
    if not winners_frame.empty:
        winners_frame["mse_vs_capacity_disagreement"] = winners_frame["best_mse_model"] != winners_frame["best_capacity_model"]
        winners_frame["accuracy_vs_latency_disagreement"] = winners_frame["best_mse_model"] != winners_frame["lowest_latency_model"]
        winners_frame["objective_disagreement"] = (
            winners_frame["mse_vs_capacity_disagreement"] | winners_frame["accuracy_vs_latency_disagreement"]
        )

    gap_rows = []
    for (pool, dataset, stress), group in stats.groupby(["pool", "dataset", "stress"], dropna=False):
        complete = group[group["complete_5_seed"]].copy()
        candidate = complete if not complete.empty else group.copy()
        if len(candidate) < 2:
            continue
        mse_ranked = candidate.sort_values(["mse_mean", "model"]).reset_index(drop=True)
        cap_ranked = candidate.sort_values(["capacity_cost_mean", "model"]).reset_index(drop=True)
        best_mse = mse_ranked.iloc[0]
        runner_mse = mse_ranked.iloc[1]
        best_cap = cap_ranked.iloc[0]
        runner_cap = cap_ranked.iloc[1]

        def upper(row: pd.Series, mean_col: str, std_col: str) -> float:
            std = 0.0 if pd.isna(row[std_col]) else float(row[std_col])
            return float(row[mean_col]) + std

        def lower(row: pd.Series, mean_col: str, std_col: str) -> float:
            std = 0.0 if pd.isna(row[std_col]) else float(row[std_col])
            return float(row[mean_col]) - std

        gap_rows.append(
            {
                "pool": pool,
                "dataset": dataset,
                "stress": stress,
                "best_mse_model": best_mse["model"],
                "second_mse_model": runner_mse["model"],
                "mse_gap": float(runner_mse["mse_mean"] - best_mse["mse_mean"]),
                "mse_gap_outside_1std": bool(
                    upper(best_mse, "mse_mean", "mse_std") < lower(runner_mse, "mse_mean", "mse_std")
                ),
                "best_capacity_model": best_cap["model"],
                "second_capacity_model": runner_cap["model"],
                "capacity_gap": float(runner_cap["capacity_cost_mean"] - best_cap["capacity_cost_mean"]),
                "capacity_gap_outside_1std": bool(
                    upper(best_cap, "capacity_cost_mean", "capacity_cost_std")
                    < lower(runner_cap, "capacity_cost_mean", "capacity_cost_std")
                ),
                "mse_vs_capacity_disagreement": bool(best_mse["model"] != best_cap["model"]),
            }
        )
    gap_frame = pd.DataFrame(gap_rows)

    rank_rows = []
    objective_cols = {
        "mse": "mse",
        "capacity": "capacity_cost",
        "latency": "latency_p95_ms",
    }
    for (pool, dataset, stress), group in frame.groupby(["pool", "dataset", "stress"], dropna=False):
        models = sorted(group["model"].unique())
        for objective, col in objective_cols.items():
            seed_winners = []
            for seed, seed_group in group.groupby("seed"):
                if len(seed_group) < 2:
                    continue
                winner = seed_group.sort_values([col, "model"]).iloc[0]["model"]
                seed_winners.append(winner)
            if not seed_winners:
                continue
            counts = pd.Series(seed_winners).value_counts()
            rank_rows.append(
                {
                    "pool": pool,
                    "dataset": dataset,
                    "stress": stress,
                    "objective": objective,
                    "seeds": len(seed_winners),
                    "models": ",".join(models),
                    "modal_winner": counts.index[0],
                    "modal_winner_count": int(counts.iloc[0]),
                    "modal_winner_rate": float(counts.iloc[0] / len(seed_winners)),
                }
            )
    rank_stability = pd.DataFrame(rank_rows)

    disagreement_rows = []
    for (pool, dataset, stress), group in frame.groupby(["pool", "dataset", "stress"], dropna=False):
        for seed, seed_group in group.groupby("seed"):
            if len(seed_group) < 2:
                continue
            best_mse = seed_group.sort_values(["mse", "model"]).iloc[0]["model"]
            best_capacity = seed_group.sort_values(["capacity_cost", "model"]).iloc[0]["model"]
            best_latency = seed_group.sort_values(["latency_p95_ms", "model"]).iloc[0]["model"]
            disagreement_rows.append(
                {
                    "pool": pool,
                    "dataset": dataset,
                    "stress": stress,
                    "seed": seed,
                    "best_mse_model": best_mse,
                    "best_capacity_model": best_capacity,
                    "lowest_latency_model": best_latency,
                    "mse_vs_capacity_disagreement": best_mse != best_capacity,
                    "accuracy_vs_latency_disagreement": best_mse != best_latency,
                    "objective_disagreement": (best_mse != best_capacity) or (best_mse != best_latency),
                }
            )
    disagreement = pd.DataFrame(disagreement_rows)
    disagreement_stability = (
        disagreement.groupby(["pool", "dataset", "stress"], dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            mse_vs_capacity_disagreement_rate=("mse_vs_capacity_disagreement", "mean"),
            accuracy_vs_latency_disagreement_rate=("accuracy_vs_latency_disagreement", "mean"),
            disagreement_rate=("objective_disagreement", "mean"),
        )
        .reset_index()
        .sort_values(["pool", "dataset", "stress"])
        if not disagreement.empty
        else pd.DataFrame()
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "main_five_seed_long.csv", index=False)
    stats.to_csv(output_dir / "main_five_seed_summary.csv", index=False)
    winners_frame.to_csv(output_dir / "main_five_seed_winners.csv", index=False)
    gap_frame.to_csv(output_dir / "gap_outside_noise.csv", index=False)
    rank_stability.to_csv(output_dir / "rank_stability.csv", index=False)
    disagreement_stability.to_csv(output_dir / "objective_disagreement_stability.csv", index=False)

    complete_core = stats[(stats["pool"] == "core_lightweight") & stats["complete_5_seed"]]
    core_expected_rows = 2 * len(SCENARIOS) * len(CORE_MODELS)
    complete_core_rows = len(complete_core)
    official_stats = stats[stats["pool"] == "official_probe"]
    core_winners = winners_frame[winners_frame["pool"] == "core_lightweight"] if not winners_frame.empty else pd.DataFrame()
    core_true_flips = int(core_winners["mse_vs_capacity_disagreement"].sum()) if not core_winners.empty else 0
    core_settings = len(core_winners)
    core_gap = gap_frame[gap_frame["pool"] == "core_lightweight"] if not gap_frame.empty else pd.DataFrame()
    hard_flip_count = int(
        (
            core_gap["mse_vs_capacity_disagreement"]
            & core_gap["capacity_gap_outside_1std"]
        ).sum()
    ) if not core_gap.empty else 0
    lines = [
        "# Five-seed decision report",
        "",
        f"Expected complete core rows: {core_expected_rows}; observed complete core rows: {complete_core_rows}.",
        f"Official probe rows present: {len(official_stats)}.",
        f"Core MSE-vs-capacity disagreements: {core_true_flips}/{core_settings}.",
        f"Core MSE-vs-capacity disagreements with capacity winner gap outside one standard deviation: {hard_flip_count}/{core_settings}.",
        "",
        "## Objective disagreement stability",
        "",
        to_markdown(disagreement_stability) if not disagreement_stability.empty else "No disagreement rows yet.",
        "",
        "Note: `objective_disagreement` combines MSE-vs-capacity disagreement and accuracy-vs-latency disagreement. "
        "The latter is often driven by the structurally low latency of DLinear and should not be used as the main ranking-reversal claim.",
        "",
        "## Winner summary",
        "",
        to_markdown(winners_frame) if not winners_frame.empty else "No winner rows yet.",
        "",
        "## Gap outside one-standard-deviation bands",
        "",
        to_markdown(gap_frame) if not gap_frame.empty else "No gap rows yet.",
        "",
    ]
    if complete_core_rows >= core_expected_rows:
        lines += [
            "## Decision",
            "",
            "Core lightweight 5-seed evidence is complete. Use MSE-vs-capacity disagreement as the primary non-trivial seed-stability defense, and keep accuracy-vs-latency disagreement as a deployment-cost observation rather than the main novelty claim.",
        ]
    else:
        lines += [
            "## Decision",
            "",
            "Core lightweight 5-seed evidence is incomplete. Continue running missing dataset/scenario/seed combinations before moving results into the paper.",
        ]
    (output_dir / "five_seed_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize five-seed AIOpsStressBench stability results.")
    parser.add_argument("--output-dir", default="outputs/five_seed")
    parser.add_argument("--expected-seeds", nargs="*", type=int, default=[42, 2024, 2025, 2026, 2027])
    parser.add_argument(
        "--core-summary",
        action="append",
        default=["outputs/five_seed/core_lightweight_summary.csv"],
        help="Core lightweight summary CSV. Can be repeated.",
    )
    parser.add_argument("--official-expanded-dir", default="outputs/five_seed/official_probe")
    args = parser.parse_args()

    core = load_core([Path(path) for path in args.core_summary])
    official = load_official(
        [
            Path("outputs/official_patchtst_native_summary.csv"),
            Path("outputs/official_patchtst_salesforce_borg_256x2048_summary.csv"),
            Path("outputs/official_itransformer_native_summary.csv"),
            Path("outputs/official_itransformer_salesforce_borg_256x2048_summary.csv"),
            Path("outputs/official_timemixer_native_summary.csv"),
            Path("outputs/official_timemixer_salesforce_borg_256x2048_summary.csv"),
        ],
        Path(args.official_expanded_dir),
    )
    frame = pd.concat([f for f in [core, official] if not f.empty], ignore_index=True)
    if frame.empty:
        raise FileNotFoundError("No five-seed input summaries found.")
    summarize(frame, Path(args.output_dir), set(args.expected_seeds))
    print(f"Saved five-seed summaries to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
