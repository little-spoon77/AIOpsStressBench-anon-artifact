from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect flat run directories into one CSV.")
    parser.add_argument("--runs", nargs="+", required=True, help="Pairs like name=outputs/path.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frames = []
    for item in args.runs:
        if "=" not in item:
            raise ValueError(f"Run must be name=path: {item}")
        name, path_text = item.split("=", 1)
        path = Path(path_text) / "metrics.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame.insert(0, "run", name)
        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.sort_values(["run", "mse"]).to_string(index=False))
    print(f"\nSaved collection to: {output.resolve()}")


if __name__ == "__main__":
    main()

