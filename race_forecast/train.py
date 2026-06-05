from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from race_forecast.config import CapacityConfig, StressConfig, TrainConfig
from race_forecast.metrics import capacity_proxy, count_parameters, measure_latency, mse_mae
from race_forecast.models import is_trainable
from race_forecast.plotting import plot_case_study
from race_forecast.stress import apply_stress


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def _predict(model: nn.Module, loader: DataLoader, stress_cfg: StressConfig, device: torch.device):
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = apply_stress(x.to(device), stress_cfg, train=False)
            pred = model(x)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def _calibrate_if_needed(
    model: nn.Module,
    name: str,
    val_loader: DataLoader,
    pred: np.ndarray,
    stress_cfg: StressConfig,
    device: torch.device,
    enabled: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    if not enabled or not name.startswith("race_"):
        return pred, {"calibration_abs_bias": 0.0, "calibration_enabled": 0.0}
    val_pred, val_true = _predict(model, val_loader, stress_cfg, device)
    correction = np.mean(val_true - val_pred, axis=0, keepdims=True)
    base_mse = float(np.mean((val_pred - val_true) ** 2))
    calibrated_mse = float(np.mean((val_pred + correction - val_true) ** 2))
    if calibrated_mse > base_mse:
        return pred, {"calibration_abs_bias": 0.0, "calibration_enabled": 0.0}
    calibrated = pred + correction
    return calibrated, {"calibration_abs_bias": float(np.mean(np.abs(correction))), "calibration_enabled": 1.0}


def _evaluate_loss(model: nn.Module, loader: DataLoader, stress_cfg: StressConfig, device: torch.device) -> float:
    pred, true = _predict(model, loader, stress_cfg, device)
    return float(np.mean((pred - true) ** 2))


def train_one_model(
    model: nn.Module,
    name: str,
    train_dataset,
    val_dataset,
    test_dataset,
    train_cfg: TrainConfig,
    stress_cfg: StressConfig,
    capacity_cfg: CapacityConfig,
    output_dir: str | Path,
    device: torch.device,
    target_mean: float = 0.0,
    target_scale: float = 1.0,
) -> dict[str, float | str | int]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    train_loader = make_loader(train_dataset, train_cfg.batch_size, True, train_cfg.num_workers)
    val_loader = make_loader(val_dataset, train_cfg.batch_size, False, train_cfg.num_workers)
    test_loader = make_loader(test_dataset, train_cfg.batch_size, False, train_cfg.num_workers)

    if is_trainable(model):
        optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
        criterion = nn.MSELoss()
        best_state = None
        best_val = float("inf")
        stale = 0
        for _epoch in range(train_cfg.epochs):
            model.train()
            for x, y in train_loader:
                x = x.to(device)
                if train_cfg.train_stress:
                    x = apply_stress(x, stress_cfg, train=True)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            val_loss = _evaluate_loss(model, val_loader, stress_cfg, device)
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= train_cfg.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)

    pred, true = _predict(model, test_loader, stress_cfg, device)
    pred, calibration_stats = _calibrate_if_needed(model, name, val_loader, pred, stress_cfg, device, train_cfg.calibrate)
    quality = mse_mae(pred, true)
    pred_raw = pred * target_scale + target_mean
    true_raw = true * target_scale + target_mean
    capacity = capacity_proxy(
        pred_raw,
        true_raw,
        headroom=capacity_cfg.headroom,
        under_cost=capacity_cfg.under_cost,
        over_cost=capacity_cfg.over_cost,
        demand_floor=capacity_cfg.demand_floor,
    )
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x[: min(32, sample_x.shape[0])]
    latency = measure_latency(
        model,
        apply_stress(sample_x, stress_cfg, train=False),
        warmup=train_cfg.latency_warmup,
        iters=train_cfg.latency_iters,
        device=device,
    )
    plot_case_study(pred_raw, true_raw, output_dir / f"{name}_case_study.png", f"{name} forecast failure window")

    result: dict[str, float | str | int] = {
        "model": name,
        "params": count_parameters(model),
        **calibration_stats,
        **quality,
        **latency,
        "capacity_under_rate": capacity.under_rate,
        "capacity_over_rate": capacity.over_rate,
        "capacity_mean_under": capacity.mean_under,
        "capacity_mean_over": capacity.mean_over,
        "capacity_cost": capacity.cost,
    }
    if device.type == "cuda":
        result["max_memory_mb"] = int(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
    else:
        result["max_memory_mb"] = 0
    return result
