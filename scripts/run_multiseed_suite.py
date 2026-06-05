from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


SCENARIOS = {
    "clean": {"scenario": "clean"},
    "missing_30": {"scenario": "missing_points", "missing_rate": 0.3},
    "missing_variables_30": {"scenario": "missing_variables", "missing_rate": 0.3},
    "delayed_12": {"scenario": "delayed_tail", "delay_steps": 12},
    "level_shift": {"scenario": "level_shift", "level_shift": 0.4},
}


DEFAULT_STRESS = {
    "scenario": "clean",
    "missing_rate": 0.0,
    "noise_std": 0.0,
    "delay_steps": 0,
    "burst_rate": 0.0,
    "level_shift": 0.0,
}


def make_stress(values: dict[str, Any]) -> dict[str, Any]:
    stress = dict(DEFAULT_STRESS)
    stress.update(values)
    return stress


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed stability experiments for AIOpsStressBench.")
    parser.add_argument("--base-config", default="configs/alibaba2018_machine_usage.yaml")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--output-root", default="outputs/multiseed")
    parser.add_argument("--summary", default="outputs/multiseed_summary.csv")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 2025, 2026])
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()))
    parser.add_argument("--models", nargs="*", default=["last_value", "dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with Path(args.base_config).open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if summary_path.exists():
        summary_frame = pd.read_csv(summary_path)
        rows = summary_frame.to_dict("records")
    else:
        summary_frame = pd.DataFrame()
        rows = []

    for seed in args.seeds:
        for scenario in args.scenarios:
            if scenario not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario}")
            if not args.force and not summary_frame.empty:
                done = summary_frame[
                    (summary_frame["dataset"] == args.dataset)
                    & (summary_frame["seed"] == seed)
                    & (summary_frame["stress"] == scenario)
                ]
                if not done.empty:
                    print(f"Skip existing seed={seed} scenario={scenario}")
                    continue

            cfg = dict(base)
            cfg["seed"] = seed
            cfg["stress"] = make_stress(SCENARIOS[scenario])
            cfg.setdefault("train", {})
            cfg["train"]["models"] = args.models
            cfg["train"]["epochs"] = args.epochs
            cfg["train"]["batch_size"] = args.batch_size
            cfg["train"]["device"] = args.device
            output_dir = Path(args.output_root) / args.dataset / f"seed_{seed}" / scenario
            cfg["output_dir"] = str(output_dir)

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)
                tmp_path = f.name
            try:
                subprocess.run([sys.executable, "-m", "race_forecast.run", "--config", tmp_path], check=True)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            metrics = pd.read_csv(output_dir / "metrics.csv")
            for row in metrics.to_dict("records"):
                rows.append(
                    {
                        "source": args.source,
                        "dataset": args.dataset,
                        "seed": seed,
                        "stress": scenario,
                        **row,
                    }
                )
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
