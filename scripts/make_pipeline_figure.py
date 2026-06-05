from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 10,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def add_box(ax, x: float, y: float, w: float, h: float, title: str, body: str, color: str) -> None:
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#2F3542",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center", weight="bold", color="#111111")
    ax.text(x + w / 2, y + h * 0.34, body, ha="center", va="center", color="#222222", linespacing=1.05)


def add_arrow(ax, x1: float, y1: float, x2: float, y2: float) -> None:
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.0,
        color="#2F3542",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the AIOpsStressBench pipeline figure.")
    parser.add_argument("--output-dir", default="outputs/paper_figures")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()
    fig, ax = plt.subplots(figsize=(7.25, 2.45))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.02, 0.48, 0.15, 0.34, "Telemetry", "public\nsources", "#E8F1FB"),
        (0.22, 0.48, 0.15, 0.34, "Audit", "entities\nsteps\nmetrics", "#F1F3F5"),
        (0.42, 0.48, 0.15, 0.34, "Stress", "missing\noutage\ndelay\nshift", "#FFF3BF"),
        (0.62, 0.48, 0.15, 0.34, "Models", "linear\npatch\nofficial", "#E6F4EA"),
        (0.82, 0.48, 0.15, 0.34, "Metrics", "accuracy\nlatency\ncapacity", "#FDECEF"),
    ]
    for item in boxes:
        add_box(ax, *item)

    for idx in range(len(boxes) - 1):
        x, y, w, h, *_ = boxes[idx]
        nx, ny, nw, nh, *_ = boxes[idx + 1]
        add_arrow(ax, x + w, y + h / 2, nx, ny + nh / 2)

    add_box(
        ax,
        0.24,
        0.08,
        0.22,
        0.23,
        "Capacity proxy",
        "forecast policy\ncontroller ref.\nunder/over cost",
        "#EDE7F6",
    )
    add_box(
        ax,
        0.54,
        0.08,
        0.22,
        0.23,
        "Case studies",
        "outage\nshift\nmissing\nresources",
        "#E0F7FA",
    )
    add_arrow(ax, 0.895, 0.48, 0.35, 0.31)
    add_arrow(ax, 0.895, 0.48, 0.65, 0.31)

    ax.text(
        0.5,
        0.94,
        "AIOpsStressBench evaluation flow",
        ha="center",
        va="center",
        fontsize=10.5,
        weight="bold",
    )

    for ext in ("png", "pdf"):
        path = output_dir / f"figure_aiopsstressbench_pipeline.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
