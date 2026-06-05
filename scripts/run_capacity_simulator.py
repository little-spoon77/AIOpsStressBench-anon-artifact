from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from race_forecast.config import StressConfig, load_config
from race_forecast.data import build_window_bundle
from race_forecast.metrics import mse_mae
from race_forecast.models import build_model
from race_forecast.stress import apply_stress
from race_forecast.train import _calibrate_if_needed, _predict, select_device, train_one_model


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stress(values: dict[str, float | str]) -> StressConfig:
    cfg = StressConfig()
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


def make_loader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)


def collect_context_and_true(loader: DataLoader, stress_cfg: StressConfig, device: torch.device, target_metric: int) -> tuple[np.ndarray, np.ndarray]:
    contexts = []
    trues = []
    with torch.no_grad():
        for x, y in loader:
            stressed = apply_stress(x.to(device), stress_cfg, train=False).detach().cpu().numpy()
            contexts.append(stressed[:, :, target_metric])
            trues.append(y.numpy())
    return np.concatenate(contexts, axis=0), np.concatenate(trues, axis=0)


def to_raw(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return values * scale + mean


def forecast_capacity(pred_raw: np.ndarray, headroom: float) -> np.ndarray:
    return np.maximum(pred_raw, 0.0) * (1.0 + headroom)


def last_observed_capacity(context_raw: np.ndarray, horizon: int, headroom: float) -> np.ndarray:
    last_seen = context_raw[:, -1:]
    return np.maximum(np.repeat(last_seen, horizon, axis=1), 0.0) * (1.0 + headroom)


def reactive_hpa_capacity(context_raw: np.ndarray, true_raw: np.ndarray, headroom: float) -> np.ndarray:
    previous = np.concatenate([context_raw[:, -1:], true_raw[:, :-1]], axis=1)
    return np.maximum(previous, 0.0) * (1.0 + headroom)


def simulate_capacity(
    capacity: np.ndarray,
    demand_raw: np.ndarray,
    under_cost: float,
    over_cost: float,
    demand_floor: float,
    severe_threshold: float,
) -> dict[str, float]:
    demand = np.maximum(demand_raw, 0.0)
    provision = np.maximum(capacity, 0.0)
    floor = max(float(demand_floor), 1e-6)
    denom = np.maximum(demand, floor)
    under = np.maximum(demand - provision, 0.0)
    over = np.maximum(provision - demand, 0.0)
    under_ratio = under / denom
    over_ratio = over / denom
    under_area = float(np.sum(under) / np.sum(np.maximum(demand, floor)))
    over_area = float(np.sum(over) / np.sum(np.maximum(demand, floor)))
    total_cost = under_cost * under_area + over_cost * over_area
    peak_miss = np.max(demand, axis=1) > np.max(provision, axis=1)
    return {
        "under_provision_rate": float(np.mean(under > 0)),
        "under_provision_area": under_area,
        "over_provision_area": over_area,
        "peak_miss_rate": float(np.mean(peak_miss)),
        "severe_under_rate": float(np.mean(under_ratio > severe_threshold)),
        "p95_under_ratio": float(np.percentile(under_ratio, 95)),
        "total_normalized_cost": float(total_cost),
    }


def policy_row(
    source: str,
    dataset: str,
    stress: str,
    model: str,
    policy: str,
    pred_raw: np.ndarray,
    true_raw: np.ndarray,
    capacity: np.ndarray,
    base_metrics: dict[str, float | int | str],
    under_cost: float,
    over_cost: float,
    demand_floor: float,
    severe_threshold: float,
) -> dict[str, float | int | str]:
    quality = mse_mae(pred_raw, true_raw)
    sim = simulate_capacity(capacity, true_raw, under_cost, over_cost, demand_floor, severe_threshold)
    return {
        "source": source,
        "dataset": dataset,
        "stress": stress,
        "model": model,
        "policy": policy,
        "mse_raw": quality["mse"],
        "mae_raw": quality["mae"],
        "latency_p95_ms": base_metrics.get("latency_p95_ms", np.nan),
        "params": base_metrics.get("params", np.nan),
        "max_memory_mb": base_metrics.get("max_memory_mb", np.nan),
        **sim,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forecast-to-capacity simulation for AIOpsStressBench.")
    parser.add_argument("--base-config", default="configs/alibaba2018_machine_usage.yaml")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--output-root", default="outputs/capacity_simulator")
    parser.add_argument("--summary", default="outputs/capacity_simulator_summary.csv")
    parser.add_argument("--models", nargs="*", default=["last_value", "dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--scenarios", nargs="*", default=["clean", "missing_30", "missing_variables_30", "delayed_12", "noisy", "burst", "level_shift"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--severe-threshold", type=float, default=0.1)
    parser.add_argument("--pred-lens", nargs="*", type=int, default=None, help="Optional horizon sensitivity values.")
    parser.add_argument("--headrooms", nargs="*", type=float, default=None, help="Optional capacity headroom sensitivity values.")
    args = parser.parse_args()

    rows = []
    base_cfg = load_config(args.base_config)
    pred_lens = args.pred_lens or [base_cfg.data.pred_len]
    headrooms = args.headrooms or [base_cfg.capacity.headroom]
    for pred_len in pred_lens:
      for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        set_seed(args.seed)
        cfg = load_config(args.base_config)
        cfg.data.pred_len = pred_len
        cfg.stress = make_stress(SCENARIOS[scenario])
        cfg.train.models = args.models
        cfg.train.epochs = args.epochs
        cfg.train.batch_size = args.batch_size
        cfg.train.device = args.device
        output_dir = Path(args.output_root) / args.dataset / f"pred_{pred_len}" / scenario
        device = select_device(cfg.train.device)
        bundle = build_window_bundle(cfg.data, args.seed)
        target_mean = float(bundle.scaler.mean_[cfg.data.target_metric])
        target_scale = float(bundle.scaler.scale_[cfg.data.target_metric])
        test_loader = make_loader(bundle.test_clean, cfg.train.batch_size, cfg.train.num_workers)
        context_scaled, true_scaled = collect_context_and_true(test_loader, cfg.stress, device, cfg.data.target_metric)
        context_raw = to_raw(context_scaled, target_mean, target_scale)
        true_raw = to_raw(true_scaled, target_mean, target_scale)

        for headroom in headrooms:
            last_cap = last_observed_capacity(context_raw, true_raw.shape[1], headroom)
            last_pred = last_cap / (1.0 + headroom)
            rows.append(
                {
                    "pred_len": pred_len,
                    "headroom": headroom,
                    **policy_row(
                        args.source,
                        args.dataset,
                        scenario,
                        "last_observed",
                        "reactive_baseline",
                        last_pred,
                        true_raw,
                        last_cap,
                        {"latency_p95_ms": 0.0, "params": 0, "max_memory_mb": 0},
                        cfg.capacity.under_cost,
                        cfg.capacity.over_cost,
                        cfg.capacity.demand_floor,
                        args.severe_threshold,
                    ),
                }
            )

            hpa_cap = reactive_hpa_capacity(context_raw, true_raw, headroom)
            hpa_pred = hpa_cap / (1.0 + headroom)
            rows.append(
                {
                    "pred_len": pred_len,
                    "headroom": headroom,
                    **policy_row(
                        args.source,
                        args.dataset,
                        scenario,
                        "reactive_hpa",
                        "reactive_baseline",
                        hpa_pred,
                        true_raw,
                        hpa_cap,
                        {"latency_p95_ms": 0.0, "params": 0, "max_memory_mb": 0},
                        cfg.capacity.under_cost,
                        cfg.capacity.over_cost,
                        cfg.capacity.demand_floor,
                        args.severe_threshold,
                    ),
                }
            )

        for model_name in args.models:
            set_seed(args.seed)
            model = build_model(
                model_name,
                input_len=cfg.data.input_len,
                pred_len=cfg.data.pred_len,
                n_metrics=len(bundle.metric_names),
                target_metric=cfg.data.target_metric,
            )
            result = train_one_model(
                model=model,
                name=model_name,
                train_dataset=bundle.train,
                val_dataset=bundle.val,
                test_dataset=bundle.test_clean,
                train_cfg=cfg.train,
                stress_cfg=cfg.stress,
                capacity_cfg=cfg.capacity,
                output_dir=output_dir / model_name,
                device=device,
                target_mean=target_mean,
                target_scale=target_scale,
            )
            val_loader = make_loader(bundle.val, cfg.train.batch_size, cfg.train.num_workers)
            pred_scaled, _ = _predict(model, test_loader, cfg.stress, device)
            pred_scaled, _ = _calibrate_if_needed(model, model_name, val_loader, pred_scaled, cfg.stress, device, cfg.train.calibrate)
            pred_raw = to_raw(pred_scaled, target_mean, target_scale)
            for headroom in headrooms:
                capacity = forecast_capacity(pred_raw, headroom)
                rows.append(
                    {
                        "pred_len": pred_len,
                        "headroom": headroom,
                        **policy_row(
                            args.source,
                            args.dataset,
                            scenario,
                            model_name,
                            "forecast_capacity",
                            pred_raw,
                            true_raw,
                            capacity,
                            result,
                            cfg.capacity.under_cost,
                            cfg.capacity.over_cost,
                            cfg.capacity.demand_floor,
                            args.severe_threshold,
                        ),
                    }
                )
            frame = pd.DataFrame(rows)
            frame.to_csv(args.summary, index=False)
            print(frame.tail(min(len(frame), 8)).to_string(index=False))

    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
