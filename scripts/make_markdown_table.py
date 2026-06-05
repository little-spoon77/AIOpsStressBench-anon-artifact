from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def to_markdown(frame: pd.DataFrame) -> str:
    headers = frame.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a compact markdown table from collected metrics.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude-models", nargs="*", default=[])
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if args.exclude_models:
        frame = frame[~frame["model"].isin(args.exclude_models)]
    cols = ["run", "model", "mse", "mae", "latency_p95_ms", "capacity_cost", "params", "max_memory_mb"]
    frame = frame[cols].copy()
    frame = frame.sort_values(["run", "mse"])
    for col in ["mse", "mae", "latency_p95_ms", "capacity_cost"]:
        frame[col] = frame[col].map(lambda x: f"{x:.4g}")
    markdown = to_markdown(frame)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    print(f"\nSaved markdown table to: {output.resolve()}")


if __name__ == "__main__":
    main()
