from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml


SCENARIOS = {
    "clean": {"scenario": "clean", "missing_rate": 0.0, "delay_steps": 0, "level_shift": 0.0},
    "missing_30": {"scenario": "missing_points", "missing_rate": 0.3, "delay_steps": 0, "level_shift": 0.0},
    "missing_variables_30": {"scenario": "missing_variables", "missing_rate": 0.3, "delay_steps": 0, "level_shift": 0.0},
    "delayed_12": {"scenario": "delayed_tail", "missing_rate": 0.0, "delay_steps": 12, "level_shift": 0.0},
    "level_shift": {"scenario": "level_shift", "missing_rate": 0.0, "delay_steps": 0, "level_shift": 0.4},
}


IMPUTATIONS = {
    "none": "none",
    "ffill": "forward_fill",
    "mean": "mean",
}


DATASETS = {
    "alibaba2018": {
        "config": "configs/alibaba2018_machine_usage.yaml",
        "epochs": 8,
    },
    "salesforce_borg": {
        "config": "configs/salesforce_borg_256x2048.yaml",
        "epochs": 6,
    },
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def run_config(cfg: dict, python: str, log_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle)
        tmp_path = handle.name
    try:
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                [python, "-m", "race_forecast.run", "--config", tmp_path],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def summarize_run(dataset: str, scenario: str, imputation_label: str, output_dir: Path) -> list[dict[str, object]]:
    metrics_path = output_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    frame = pd.read_csv(metrics_path)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            {
                "source": dataset,
                "stress": scenario,
                "imputation": imputation_label,
                "pipeline": f"{row['model']}+{imputation_label}",
                "model": row["model"],
                "mse": row["mse"],
                "mae": row["mae"],
                "latency_p95_ms": row.get("latency_p95_ms"),
                "max_memory_mb": row.get("max_memory_mb"),
                "capacity_cost": row.get("capacity_cost"),
                "capacity_under_rate": row.get("capacity_under_rate"),
                "capacity_over_rate": row.get("capacity_over_rate"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight imputation + forecasting baselines.")
    parser.add_argument("--datasets", nargs="*", default=["alibaba2018", "salesforce_borg"], choices=DATASETS)
    parser.add_argument("--scenarios", nargs="*", default=list(SCENARIOS.keys()), choices=SCENARIOS)
    parser.add_argument("--imputations", nargs="*", default=["none", "ffill", "mean"], choices=IMPUTATIONS)
    parser.add_argument("--models", nargs="*", default=["dlinear", "patchtst"], choices=["dlinear", "patchtst"])
    parser.add_argument("--output-root", default="outputs/imputation_pipeline")
    parser.add_argument("--summary", default="outputs/imputation_pipeline_summary.csv")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true", help="Reuse existing metrics.csv files and run only missing combinations.")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, object]] = []

    for dataset in args.datasets:
        dataset_info = DATASETS[dataset]
        base = load_config(Path(dataset_info["config"]))
        base.setdefault("train", {})
        base["train"]["models"] = args.models
        base["train"]["epochs"] = dataset_info["epochs"]
        if args.device is not None:
            base["train"]["device"] = args.device

        for scenario_name in args.scenarios:
            for imputation_label in args.imputations:
                cfg = dict(base)
                cfg["data"] = dict(base["data"])
                cfg["train"] = dict(base["train"])
                cfg["capacity"] = dict(base.get("capacity", {}))
                stress = {
                    "scenario": "clean",
                    "missing_rate": 0.0,
                    "noise_std": 0.0,
                    "delay_steps": 0,
                    "burst_rate": 0.0,
                    "level_shift": 0.0,
                    "imputation": IMPUTATIONS[imputation_label],
                }
                stress.update(SCENARIOS[scenario_name])
                stress["imputation"] = IMPUTATIONS[imputation_label]
                cfg["stress"] = stress
                cfg["output_dir"] = str(output_root / dataset / scenario_name / imputation_label)
                run_output = Path(cfg["output_dir"])
                metrics_path = run_output / "metrics.csv"
                if not (args.resume and metrics_path.exists()):
                    run_output.mkdir(parents=True, exist_ok=True)
                    run_config(cfg, args.python, run_output / "run.log")
                summary_rows.extend(summarize_run(dataset, scenario_name, imputation_label, run_output))

    summary = pd.DataFrame(summary_rows)
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
