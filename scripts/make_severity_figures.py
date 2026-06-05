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


FAMILY_TITLES = {
    "missing_points": "Point missingness",
    "missing_variables": "Variable outage",
    "delayed_tail": "Telemetry delay",
    "noise": "Sensor noise",
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


def add_relative_mse(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    baselines = (
        frame[frame["level"].astype(float) == 0.0][["source", "dataset", "stress_family", "model", "mse"]]
        .rename(columns={"mse": "clean_mse"})
    )
    frame = frame.merge(baselines, on=["source", "dataset", "stress_family", "model"], how="left")
    frame["relative_mse"] = frame["mse"] / frame["clean_mse"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stress-severity curve figures.")
    parser.add_argument("--summary", default="outputs/severity_curve_summary.csv")
    parser.add_argument("--output-dir", default="outputs/paper_figures")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--models", nargs="*", default=["last_value", "dlinear", "race_dlinear", "patchtst"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.summary)
    frame = frame[(frame["source"] == args.source) & (frame["dataset"] == args.dataset) & frame["model"].isin(args.models)].copy()
    if frame.empty:
        raise SystemExit("No severity rows matched the requested source/dataset/models.")
    frame["level"] = frame["level"].astype(float)
    frame = add_relative_mse(frame)
    source_csv = output_dir / f"figure_stress_severity_curves_{args.dataset}.csv"
    frame.to_csv(source_csv, index=False)
    print(f"Saved {source_csv}")

    apply_style()
    families = [family for family in ["missing_points", "missing_variables", "delayed_tail", "noise"] if family in set(frame["stress_family"])]
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 5.8), constrained_layout=True)
    axes = axes.flatten()
    for ax, family in zip(axes, families):
        sub = frame[frame["stress_family"] == family].copy()
        for model in args.models:
            model_rows = sub[sub["model"] == model].sort_values("level")
            if model_rows.empty:
                continue
            ax.plot(
                model_rows["level"],
                model_rows["relative_mse"],
                marker=MODEL_MARKERS.get(model, "o"),
                color=MODEL_COLORS.get(model, "#333333"),
                linewidth=1.8,
                markersize=5,
                label=MODEL_LABELS.get(model, model),
            )
        ax.axhline(1.0, color="#999999", linewidth=0.8, linestyle="--")
        ax.set_title(FAMILY_TITLES.get(family, family))
        ax.set_xlabel("Stress level")
        ax.set_ylabel("MSE / clean MSE")
        ax.margins(x=0.05, y=0.16)
    for ax in axes[len(families) :]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False, bbox_to_anchor=(0.5, 1.04))
    for ext in ["png", "pdf"]:
        path = output_dir / f"figure_stress_severity_curves_{args.dataset}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
