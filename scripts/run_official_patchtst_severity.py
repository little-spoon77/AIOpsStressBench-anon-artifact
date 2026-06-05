from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


FAMILY_SCENARIOS = {
    "missing_points": {
        0.0: "clean",
        0.1: "missing_10",
        0.3: "missing_30",
        0.5: "missing_50",
        0.7: "missing_70",
    },
    "missing_variables": {
        0.0: "clean",
        0.1: "missing_variables_10",
        0.3: "missing_variables_30",
        0.5: "missing_variables_50",
    },
    "delayed_tail": {
        0.0: "clean",
        6.0: "delayed_6",
        12.0: "delayed_12",
        24.0: "delayed_24",
    },
    "noise": {
        0.0: "clean",
        0.1: "noise_10",
        0.2: "noisy",
        0.4: "noise_40",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert existing official native baseline runs into severity rows.")
    parser.add_argument("--official-summary", default="outputs/official_patchtst_native_summary.csv")
    parser.add_argument("--output", default="outputs/official_patchtst_severity_summary.csv")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--model", default=None, help="Optional model name override.")
    args = parser.parse_args()

    frame = pd.read_csv(args.official_summary)
    rows = []
    for family, levels in FAMILY_SCENARIOS.items():
        for level, scenario in levels.items():
            match = frame[(frame["source"] == args.source) & (frame["dataset"] == args.dataset) & (frame["stress"] == scenario)].copy()
            if match.empty:
                continue
            for row in match.to_dict("records"):
                if args.model:
                    row["model"] = args.model
                rows.append(
                    {
                        "source": args.source,
                        "dataset": args.dataset,
                        "stress_family": family,
                        "level": level,
                        "level_label": f"level_{int(level * 100) if level < 1 else int(level):02d}",
                        "stress": scenario,
                        **row,
                    }
                )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
