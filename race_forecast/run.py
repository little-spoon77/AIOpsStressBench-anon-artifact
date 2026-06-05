from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from race_forecast.config import load_config
from race_forecast.data import build_window_bundle
from race_forecast.models import build_model
from race_forecast.train import select_device, train_one_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RACE-Forecast experiments.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(cfg.train.device)
    bundle = build_window_bundle(cfg.data, cfg.seed)
    n_metrics = len(bundle.metric_names)
    print(f"Device: {device}")
    print(f"Metrics: {bundle.metric_names}")
    print(f"Train/val/test windows: {len(bundle.train)}/{len(bundle.val)}/{len(bundle.test_clean)}")
    print(f"Stress scenario: {cfg.stress.scenario}")

    results = []
    for model_name in cfg.train.models:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(
            model_name,
            input_len=cfg.data.input_len,
            pred_len=cfg.data.pred_len,
            n_metrics=n_metrics,
            target_metric=cfg.data.target_metric,
        )
        print(f"Running model: {model_name}")
        result = train_one_model(
            model=model,
            name=model_name,
            train_dataset=bundle.train,
            val_dataset=bundle.val,
            test_dataset=bundle.test_clean,
            train_cfg=cfg.train,
            stress_cfg=cfg.stress,
            capacity_cfg=cfg.capacity,
            output_dir=output_dir,
            device=device,
            target_mean=float(bundle.scaler.mean_[cfg.data.target_metric]),
            target_scale=float(bundle.scaler.scale_[cfg.data.target_metric]),
        )
        results.append(result)
        print(json.dumps(result, indent=2))

    frame = pd.DataFrame(results).sort_values(["mse", "latency_p95_ms"])
    frame.to_csv(output_dir / "metrics.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSummary:")
    print(frame.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
