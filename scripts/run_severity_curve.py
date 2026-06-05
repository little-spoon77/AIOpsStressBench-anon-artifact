from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_STRESS = {
    "scenario": "clean",
    "missing_rate": 0.0,
    "noise_std": 0.0,
    "delay_steps": 0,
    "burst_rate": 0.0,
    "level_shift": 0.0,
}


FAMILY_LEVELS = {
    "missing_points": [0.0, 0.1, 0.3, 0.5, 0.7],
    "missing_variables": [0.0, 0.1, 0.3, 0.5],
    "delayed_tail": [0, 6, 12, 24],
    "noise": [0.0, 0.1, 0.2, 0.4],
}


def stress_for(family: str, level: float) -> dict[str, Any]:
    stress = dict(DEFAULT_STRESS)
    if level == 0:
        return stress
    if family == "missing_points":
        stress.update({"scenario": "missing_points", "missing_rate": float(level)})
    elif family == "missing_variables":
        stress.update({"scenario": "missing_variables", "missing_rate": float(level)})
    elif family == "delayed_tail":
        stress.update({"scenario": "delayed_tail", "delay_steps": int(level)})
    elif family == "noise":
        stress.update({"scenario": "noisy", "noise_std": float(level)})
    else:
        raise ValueError(f"Unknown stress family: {family}")
    return stress


def level_slug(level: float) -> str:
    if float(level).is_integer():
        return str(int(level)).zfill(2)
    return str(int(round(float(level) * 100))).zfill(2)


def load_existing(summary: Path) -> pd.DataFrame:
    if summary.exists():
        return pd.read_csv(summary)
    return pd.DataFrame()


def already_done(summary_frame: pd.DataFrame, dataset: str, family: str, level: float) -> bool:
    if summary_frame.empty:
        return False
    key = summary_frame[
        (summary_frame["dataset"] == dataset)
        & (summary_frame["stress_family"] == family)
        & (summary_frame["level"].astype(float) == float(level))
    ]
    return not key.empty


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stress-severity curves for AIOpsStressBench.")
    parser.add_argument("--base-config", default="configs/alibaba2018_machine_usage.yaml")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--output-root", default="outputs/severity_curve")
    parser.add_argument("--summary", default="outputs/severity_curve_summary.csv")
    parser.add_argument("--families", nargs="*", default=["missing_points", "missing_variables", "delayed_tail", "noise"])
    parser.add_argument("--models", nargs="*", default=["last_value", "dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="Re-run levels even if they already exist in the summary.")
    args = parser.parse_args()

    with Path(args.base_config).open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_frame = load_existing(summary_path)
    rows = [] if summary_frame.empty else summary_frame.to_dict("records")

    for family in args.families:
        if family not in FAMILY_LEVELS:
            raise ValueError(f"Unknown family {family}; choose from {sorted(FAMILY_LEVELS)}")
        for level in FAMILY_LEVELS[family]:
            if not args.force and already_done(summary_frame, args.dataset, family, level):
                print(f"Skip existing {args.dataset}/{family}/{level}")
                continue

            cfg = dict(base)
            cfg["seed"] = args.seed
            cfg["stress"] = stress_for(family, level)
            cfg.setdefault("train", {})
            cfg["train"]["models"] = args.models
            cfg["train"]["epochs"] = args.epochs
            cfg["train"]["batch_size"] = args.batch_size
            cfg["train"]["device"] = args.device

            level_name = f"level_{level_slug(float(level))}"
            output_dir = Path(args.output_root) / args.dataset / family / level_name
            cfg["output_dir"] = str(output_dir)

            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)
                tmp_path = f.name
            try:
                subprocess.run([sys.executable, "-m", "race_forecast.run", "--config", tmp_path], check=True)
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            metrics_path = output_dir / "metrics.csv"
            metrics = pd.read_csv(metrics_path)
            for row in metrics.to_dict("records"):
                rows.append(
                    {
                        "source": args.source,
                        "dataset": args.dataset,
                        "stress_family": family,
                        "level": float(level),
                        "level_label": level_name,
                        "stress": cfg["stress"]["scenario"],
                        **row,
                    }
                )
            pd.DataFrame(rows).to_csv(summary_path, index=False)
            print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
