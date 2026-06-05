from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


BOXES = [
    {
        "id": "telemetry",
        "label": "Telemetry window",
        "detail": "CPU / memory / network / KPI",
        "x": 0.05,
        "y": 0.58,
        "w": 0.18,
        "h": 0.18,
    },
    {
        "id": "features",
        "label": "Stress feature extractor",
        "detail": "missing, outage, delay,\nvolatility, trend",
        "x": 0.28,
        "y": 0.58,
        "w": 0.21,
        "h": 0.18,
    },
    {
        "id": "controls",
        "label": "Deployment controls",
        "detail": "latency budget\nobjective",
        "x": 0.28,
        "y": 0.22,
        "w": 0.21,
        "h": 0.16,
    },
    {
        "id": "router",
        "label": "StressRoute policy",
        "detail": "v1 rules + v2 learned router\n+ oracle upper bound",
        "x": 0.55,
        "y": 0.48,
        "w": 0.22,
        "h": 0.22,
    },
    {
        "id": "forecaster",
        "label": "Selected forecaster",
        "detail": "DLinear / RACE-DLinear /\nPatchTST-lite",
        "x": 0.82,
        "y": 0.58,
        "w": 0.17,
        "h": 0.18,
    },
    {
        "id": "decision",
        "label": "Deployment evaluation",
        "detail": "forecast quality, latency,\ncapacity risk",
        "x": 0.82,
        "y": 0.22,
        "w": 0.17,
        "h": 0.17,
    },
]


ARROWS = [
    ("telemetry", "features"),
    ("features", "router"),
    ("controls", "router"),
    ("router", "forecaster"),
    ("forecaster", "decision"),
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def center(box: dict[str, object]) -> tuple[float, float]:
    return float(box["x"]) + float(box["w"]) / 2, float(box["y"]) + float(box["h"]) / 2


def edge_point(source: dict[str, object], target: dict[str, object]) -> tuple[tuple[float, float], tuple[float, float]]:
    sx, sy = center(source)
    tx, ty = center(target)
    sw, sh = float(source["w"]), float(source["h"])
    tw, th = float(target["w"]), float(target["h"])
    if abs(tx - sx) >= abs(ty - sy):
        start = (sx + (sw / 2) * (1 if tx > sx else -1), sy)
        end = (tx - (tw / 2) * (1 if tx > sx else -1), ty)
    else:
        start = (sx, sy + (sh / 2) * (1 if ty > sy else -1))
        end = (tx, ty - (th / 2) * (1 if ty > sy else -1))
    return start, end


def draw_box(ax, box: dict[str, object], color: str) -> None:
    patch = FancyBboxPatch(
        (float(box["x"]), float(box["y"])),
        float(box["w"]),
        float(box["h"]),
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.1,
        edgecolor=color,
        facecolor="#F8FAFC",
    )
    ax.add_patch(patch)
    ax.text(
        float(box["x"]) + float(box["w"]) / 2,
        float(box["y"]) + float(box["h"]) * 0.64,
        str(box["label"]),
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#1F2937",
    )
    ax.text(
        float(box["x"]) + float(box["w"]) / 2,
        float(box["y"]) + float(box["h"]) * 0.33,
        str(box["detail"]),
        ha="center",
        va="center",
        fontsize=7.5,
        color="#4B5563",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the StressRoute system figure.")
    parser.add_argument("--output-dir", default="outputs/paper_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(BOXES).to_csv(output_dir / "figure_stressroute_system.csv", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(9.0, 3.3))
    ax.set_xlim(0, 1.03)
    ax.set_ylim(0.08, 0.84)
    ax.axis("off")
    colors = {
        "telemetry": "#3B6FB6",
        "features": "#2F9E6D",
        "controls": "#D9822B",
        "router": "#6C5CE7",
        "forecaster": "#3B6FB6",
        "decision": "#2F9E6D",
    }
    by_id = {box["id"]: box for box in BOXES}
    for source_id, target_id in ARROWS:
        start, end = edge_point(by_id[source_id], by_id[target_id])
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color="#6B7280",
            shrinkA=3,
            shrinkB=3,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)
    for box in BOXES:
        draw_box(ax, box, colors[str(box["id"])])
    ax.text(
        0.55,
        0.14,
        "Policy output is an auditable model choice, not a production autoscaling action.",
        ha="center",
        va="center",
        fontsize=8,
        color="#6B7280",
    )
    for ext in ["png", "pdf"]:
        path = output_dir / f"figure_stressroute_system.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
