from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="List generated case-study figures for paper inspection.")
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--output", default="outputs/case_study_index.csv")
    args = parser.parse_args()

    rows = []
    for path in sorted(Path(args.root).glob("**/*_case_study.png")):
        parts = path.parts
        model = path.name.replace("_case_study.png", "")
        rows.append({"path": str(path), "model": model, "run": "/".join(parts[1:-1]) if len(parts) > 2 else path.parent.name})
    if not rows:
        raise SystemExit(f"No case study PNG files found under {args.root}")
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(frame.to_string(index=False))
    print(f"\nSaved case-study index to: {output.resolve()}")


if __name__ == "__main__":
    main()

