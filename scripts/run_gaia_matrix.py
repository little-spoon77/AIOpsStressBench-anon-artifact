from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


DATASETS = {
    "all": "data/gaia_metric_forecast.npz",
    "periodic": "data/gaia_periodic.npz",
    "changepoint": "data/gaia_changepoint.npz",
    "low_snr": "data/gaia_low_snr.npz",
    "partially_stationary": "data/gaia_partially_stationary.npz",
}

STRESSES = {
    "clean": {"scenario": "clean", "missing_rate": 0.0, "noise_std": 0.0, "delay_steps": 0, "burst_rate": 0.0, "level_shift": 0.0},
    "missing_10": {"scenario": "missing_points", "missing_rate": 0.1, "noise_std": 0.0, "delay_steps": 0, "burst_rate": 0.0, "level_shift": 0.0},
    "missing_30": {"scenario": "missing_points", "missing_rate": 0.3, "noise_std": 0.0, "delay_steps": 0, "burst_rate": 0.0, "level_shift": 0.0},
    "missing_50": {"scenario": "missing_points", "missing_rate": 0.5, "noise_std": 0.0, "delay_steps": 0, "burst_rate": 0.0, "level_shift": 0.0},
    "missing_variables_30": {"scenario": "missing_variables", "missing_rate": 0.3, "noise_std": 0.0, "delay_steps": 0, "burst_rate": 0.0, "level_shift": 0.0},
    "delayed_12": {"scenario": "delayed_tail", "missing_rate": 0.0, "noise_std": 0.0, "delay_steps": 12, "burst_rate": 0.0, "level_shift": 0.0},
    "burst": {"scenario": "burst", "missing_rate": 0.0, "noise_std": 0.0, "delay_steps": 0, "burst_rate": 0.02, "level_shift": 0.0},
    "level_shift": {"scenario": "level_shift", "missing_rate": 0.0, "noise_std": 0.0, "delay_steps": 0, "burst_rate": 0.0, "level_shift": 0.4},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAIA category x deployment-stress matrix.")
    parser.add_argument("--base-config", default="configs/gaia_metric_example.yaml")
    parser.add_argument("--output-root", default="outputs/gaia_matrix")
    parser.add_argument("--datasets", nargs="*", default=["periodic", "changepoint", "low_snr", "partially_stationary"])
    parser.add_argument("--stresses", nargs="*", default=["clean", "missing_10", "missing_30", "missing_50", "missing_variables_30", "delayed_12", "burst", "level_shift"])
    parser.add_argument("--models", nargs="*", default=["last_value", "dlinear", "race_dlinear", "patchtst"])
    args = parser.parse_args()

    with Path(args.base_config).open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        for stress_name in args.stresses:
            if stress_name not in STRESSES:
                raise ValueError(f"Unknown stress: {stress_name}")
            cfg = dict(base)
            cfg["data"] = dict(base["data"])
            cfg["data"]["npz_path"] = DATASETS[dataset_name]
            cfg["stress"] = dict(STRESSES[stress_name])
            cfg["train"] = dict(base["train"])
            cfg["train"]["models"] = args.models
            cfg["output_dir"] = str(Path(args.output_root) / dataset_name / stress_name)
            with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)
                tmp_path = f.name
            try:
                subprocess.run([sys.executable, "-m", "race_forecast.run", "--config", tmp_path], check=True)
            finally:
                Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

