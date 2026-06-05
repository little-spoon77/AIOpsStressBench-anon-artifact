from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "last_value": "LastValue",
    "dlinear": "DLinear",
    "race_dlinear": "RACE-DLinear",
    "patchtst": "PatchTST-lite",
    "official_patchtst": "Official PatchTST",
}

MODEL_COLORS = {
    "last_value": "#777777",
    "dlinear": "#3B6FB6",
    "race_dlinear": "#D9822B",
    "patchtst": "#2F9E6D",
    "official_patchtst": "#9B59B6",
}

MODEL_MARKERS = {
    "last_value": "o",
    "dlinear": "s",
    "race_dlinear": "D",
    "patchtst": "^",
    "official_patchtst": "P",
}

MODEL_LINESTYLES = {
    "last_value": ":",
    "dlinear": "-",
    "race_dlinear": "--",
    "patchtst": "-.",
    "official_patchtst": (0, (3, 1, 1, 1)),
}

SOURCE_LABELS = {
    "alibaba2018": "Alibaba 2018",
    "salesforce_borg": "Salesforce/Borg",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def add_relative_mse(frame: pd.DataFrame) -> pd.DataFrame:
    clean = (
        frame[frame["level"].astype(float) == 0.0][["source", "dataset", "stress_family", "model", "mse"]]
        .rename(columns={"mse": "clean_mse"})
    )
    frame = frame.merge(clean, on=["source", "dataset", "stress_family", "model"], how="left")
    frame["relative_mse"] = frame["mse"] / frame["clean_mse"]
    return frame


def plot_severity(ax, frame: pd.DataFrame, source: str, title: str) -> None:
    sub = frame[(frame["source"] == source) & (frame["stress_family"] == "missing_variables")].copy()
    for model in ["dlinear", "race_dlinear", "patchtst", "official_patchtst"]:
        rows = sub[sub["model"] == model].sort_values("level")
        if rows.empty:
            continue
        ax.plot(
            rows["level"],
            rows["relative_mse"],
            label=MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#333333"),
            marker=MODEL_MARKERS.get(model, "o"),
            linestyle=MODEL_LINESTYLES.get(model, "-"),
            linewidth=1.7,
            markersize=4.5,
        )
    ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Missing-variable rate")
    ax.set_ylabel("MSE / clean MSE")
    ax.margins(x=0.06, y=0.16)


def plot_capacity_pareto(ax, frame: pd.DataFrame) -> None:
    selected = frame[
        (frame["policy"] == "forecast_capacity")
        & (frame["stress"].isin(["missing_30", "missing_variables_30", "delayed_12"]))
        & (frame["model"].isin(["dlinear", "race_dlinear", "patchtst"]))
    ].copy()
    summary = (
        selected.groupby(["source", "model"], as_index=False)
        .agg(capacity_cost=("total_normalized_cost", "mean"), p95_ms=("latency_p95_ms", "mean"))
    )
    for source, source_rows in summary.groupby("source"):
        for _, row in source_rows.iterrows():
            model = row["model"]
            ax.scatter(
                row["p95_ms"],
                row["capacity_cost"],
                s=80,
                marker=MODEL_MARKERS.get(model, "o"),
                color=MODEL_COLORS.get(model, "#333333"),
                edgecolor="white",
                linewidth=0.7,
                alpha=0.92,
            )
            ax.annotate(
                f"{SOURCE_LABELS.get(source, source).split()[0]}-{MODEL_LABELS.get(model, model)}",
                (row["p95_ms"], row["capacity_cost"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.8,
            )
    ax.set_xscale("log")
    ax.set_title("Capacity cost vs latency")
    ax.set_xlabel("P95 latency (ms, log)")
    ax.set_ylabel("Mean capacity cost")
    ax.margins(x=0.18, y=0.20)


def plot_winner_disagreement(ax, winners: pd.DataFrame) -> pd.DataFrame:
    if winners.empty:
        ax.text(0.5, 0.5, "Winner table not available", ha="center", va="center")
        ax.axis("off")
        return pd.DataFrame()
    selected = winners[
        winners["source"].isin(SOURCE_LABELS)
        & winners["stress"].isin(["clean", "delayed_12", "missing_30", "missing_variables_30"])
    ].copy()
    rows = []
    pairs = [
        ("MSE vs latency", "best_mse_model", "best_latency_model"),
        ("MSE vs decision", "best_mse_model", "best_capacity_model"),
        ("Latency vs decision", "best_latency_model", "best_capacity_model"),
    ]
    for label, left, right in pairs:
        rows.append({"comparison": label, "count": int((selected[left] != selected[right]).sum())})
    summary = pd.DataFrame(rows)
    bars = ax.bar(
        summary["comparison"],
        summary["count"],
        color=["#3B6FB6", "#2F9E6D", "#D9822B"],
        width=0.62,
    )
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{int(bar.get_height())}/{len(selected)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylim(0, max(len(selected), 1) + 1)
    ax.set_ylabel("Disagreement count")
    ax.set_title("Objective-winner disagreement")
    ax.tick_params(axis="x", rotation=12)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a multi-source deployment-stress paper figure.")
    parser.add_argument("--severity", nargs="+", required=True)
    parser.add_argument("--capacity", nargs="+", required=True)
    parser.add_argument("--winners", default="outputs/paper_tables/table_scenario_winners.csv")
    parser.add_argument("--stressroute-selection", nargs="*", default=[])
    parser.add_argument("--output-dir", default="outputs/paper_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    severity = pd.concat([pd.read_csv(path) for path in args.severity], ignore_index=True)
    severity["level"] = severity["level"].astype(float)
    severity = add_relative_mse(severity)
    capacity = pd.concat([pd.read_csv(path) for path in args.capacity], ignore_index=True)
    selection_frames = [pd.read_csv(path) for path in args.stressroute_selection if Path(path).exists()]
    selection = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    winners = pd.read_csv(args.winners) if Path(args.winners).exists() else pd.DataFrame()
    disagreement = pd.DataFrame()

    source_csv = output_dir / "figure_multisource_deployment_stress.csv"
    figure_source = pd.concat(
        [
            severity.assign(panel="severity"),
            capacity.assign(panel="capacity"),
            selection.assign(panel="selection") if not selection.empty else pd.DataFrame(),
            winners.assign(panel="winner_table") if not winners.empty else pd.DataFrame(),
        ],
        ignore_index=True,
        sort=False,
    )
    figure_source.to_csv(source_csv, index=False)
    print(f"Saved {source_csv}")

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), constrained_layout=True)
    plot_severity(axes[0, 0], severity, "alibaba2018", "Alibaba metric outage")
    plot_severity(axes[0, 1], severity, "salesforce_borg", "Salesforce/Borg metric outage")
    plot_capacity_pareto(axes[1, 0], capacity)
    disagreement = plot_winner_disagreement(axes[1, 1], winners)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)), frameon=False, bbox_to_anchor=(0.5, 1.04))
    for ext in ["png", "pdf"]:
        path = output_dir / f"figure_multisource_deployment_stress.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved {path}")
    plt.close(fig)
    if not disagreement.empty:
        disagreement_path = output_dir / "figure_multisource_winner_disagreement.csv"
        disagreement.to_csv(disagreement_path, index=False)
        print(f"Saved {disagreement_path}")


if __name__ == "__main__":
    main()
