from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_LABELS = {
    "last_value": "Last value",
    "dlinear": "DLinear",
    "race_dlinear": "RACE-DLinear",
    "patchtst": "PatchTST-lite",
    "official_patchtst": "Official PatchTST",
    "official_itransformer": "Official iTransformer",
}


MODEL_COLORS = {
    "last_value": "#777777",
    "dlinear": "#3B6FB6",
    "race_dlinear": "#D9822B",
    "patchtst": "#2F9E6D",
    "official_patchtst": "#9B59B6",
    "official_itransformer": "#C0392B",
}


MODEL_MARKERS = {
    "last_value": "o",
    "dlinear": "s",
    "race_dlinear": "D",
    "patchtst": "^",
    "official_patchtst": "P",
    "official_itransformer": "X",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
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


def plot_panel(ax, frame: pd.DataFrame, title: str, y_col: str, y_label: str) -> None:
    for _, row in frame.iterrows():
        model = str(row["model"])
        ax.scatter(
            row["latency_p95_ms"],
            row[y_col],
            s=70,
            marker=MODEL_MARKERS.get(model, "o"),
            color=MODEL_COLORS.get(model, "#333333"),
            edgecolor="white",
            linewidth=0.7,
            label=MODEL_LABELS.get(model, model),
            zorder=3,
        )
        ax.annotate(
            MODEL_LABELS.get(model, model),
            (row["latency_p95_ms"], row[y_col]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.4,
            alpha=0.92,
        )
    ax.set_xscale("log")
    ax.set_xlabel("P95 latency (ms, log scale)")
    ax.set_ylabel(y_label)
    ax.set_title(title)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready accuracy-latency tradeoff figures.")
    parser.add_argument("--table-dir", default="outputs/paper_tables")
    parser.add_argument("--output-dir", default="outputs/paper_figures")
    parser.add_argument("--source", default="alibaba2018")
    args = parser.parse_args()

    table_dir = Path(args.table_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(table_dir / "table_clean_accuracy.csv")
    stress = pd.read_csv(table_dir / "table_stress_robustness.csv")
    clean = clean[clean["source"] == args.source].copy()
    stress = stress[stress["source"] == args.source].copy()
    stress = stress.rename(columns={"mean_mse": "mse", "mean_p95_ms": "latency_p95_ms"})

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4), constrained_layout=True)
    plot_panel(axes[0], clean.sort_values("latency_p95_ms"), f"{args.source}: clean", "mse", "MSE")
    plot_panel(axes[1], stress.sort_values("latency_p95_ms"), f"{args.source}: deployment stress", "mse", "Mean stress MSE")

    for ax in axes:
        ax.margins(x=0.16, y=0.18)
    fig.suptitle("Accuracy-latency tradeoff under CloudOps deployment constraints", y=1.05, fontsize=11)
    for ext in ["png", "pdf"]:
        path = output_dir / f"figure_accuracy_latency_tradeoff_{args.source}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
