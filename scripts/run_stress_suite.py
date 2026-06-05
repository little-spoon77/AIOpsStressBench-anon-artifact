from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


DEFAULT_STRESS = {
    "scenario": "clean",
    "missing_rate": 0.0,
    "noise_std": 0.0,
    "delay_steps": 0,
    "burst_rate": 0.0,
    "level_shift": 0.0,
}


SCENARIOS = {
    "clean": {"scenario": "clean"},
    "missing_10": {"scenario": "missing_points", "missing_rate": 0.1},
    "missing_30": {"scenario": "missing_points", "missing_rate": 0.3},
    "missing_50": {"scenario": "missing_points", "missing_rate": 0.5},
    "missing_variables_30": {"scenario": "missing_variables", "missing_rate": 0.3},
    "delayed_12": {"scenario": "delayed_tail", "delay_steps": 12},
    "noisy": {"scenario": "noisy", "noise_std": 0.2},
    "burst": {"scenario": "burst", "burst_rate": 0.02},
    "level_shift": {"scenario": "level_shift", "level_shift": 0.4},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deployment stress scenarios from one base config.")
    parser.add_argument("--base-config", default="configs/quick_synthetic.yaml")
    parser.add_argument("--output-root", default="outputs/stress_suite")
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()))
    args = parser.parse_args()

    with Path(args.base_config).open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    for scenario_name in args.scenarios:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        cfg = dict(base)
        cfg["stress"] = dict(DEFAULT_STRESS)
        cfg["stress"].update(SCENARIOS[scenario_name])
        cfg["output_dir"] = str(Path(args.output_root) / scenario_name)
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
            yaml.safe_dump(cfg, f)
            tmp_path = f.name
        try:
            subprocess.run([sys.executable, "-m", "race_forecast.run", "--config", tmp_path], check=True)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
