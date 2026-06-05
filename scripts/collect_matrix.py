from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GAIA matrix results into one CSV.")
    parser.add_argument("--root", default="outputs/gaia_matrix")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    frames = []
    for metrics_path in sorted(root.glob("*/*/metrics.csv")):
        dataset = metrics_path.parent.parent.name
        stress = metrics_path.parent.name
        frame = pd.read_csv(metrics_path)
        frame.insert(0, "dataset", dataset)
        frame.insert(1, "stress", stress)
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No metrics.csv files found under {root}")
    result = pd.concat(frames, ignore_index=True)
    output = Path(args.output) if args.output else root / "matrix_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.sort_values(["dataset", "stress", "mse"]).to_string(index=False))
    print(f"\nSaved matrix summary to: {output.resolve()}")


if __name__ == "__main__":
    main()

