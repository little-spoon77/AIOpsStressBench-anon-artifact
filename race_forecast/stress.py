from __future__ import annotations

import torch

from race_forecast.config import StressConfig


def _forward_fill_zeros(x: torch.Tensor) -> torch.Tensor:
    filled = x.clone()
    observed = filled.abs() > 1e-6
    for step in range(1, filled.shape[1]):
        missing = ~observed[:, step, :]
        filled[:, step, :] = torch.where(missing, filled[:, step - 1, :], filled[:, step, :])
    return filled


def _mean_fill_zeros(x: torch.Tensor) -> torch.Tensor:
    observed = x.abs() > 1e-6
    counts = observed.sum(dim=1, keepdim=True).clamp_min(1)
    means = (x * observed.float()).sum(dim=1, keepdim=True) / counts
    return torch.where(observed, x, means)


def apply_imputation(x: torch.Tensor, method: str) -> torch.Tensor:
    if method in {"none", "", None}:
        return x
    if method == "forward_fill":
        return _forward_fill_zeros(x)
    if method == "mean":
        return _mean_fill_zeros(x)
    raise ValueError(f"Unsupported imputation method: {method}")


def apply_stress(x: torch.Tensor, cfg: StressConfig, train: bool = False) -> torch.Tensor:
    if cfg.scenario == "clean":
        return apply_imputation(x, cfg.imputation)
    stressed = x.clone()

    if cfg.scenario in {"missing_points", "mixed"} and cfg.missing_rate > 0:
        mask = torch.rand_like(stressed) < cfg.missing_rate
        stressed = stressed.masked_fill(mask, 0.0)

    if cfg.scenario in {"missing_variables", "mixed"} and cfg.missing_rate > 0:
        batch, _, metrics = stressed.shape
        var_mask = torch.rand(batch, 1, metrics, device=stressed.device) < cfg.missing_rate
        stressed = stressed.masked_fill(var_mask, 0.0)

    if cfg.scenario in {"delayed_tail", "mixed"} and cfg.delay_steps > 0:
        delay = min(cfg.delay_steps, stressed.shape[1])
        stressed[:, -delay:, :] = 0.0

    if cfg.scenario in {"noisy", "mixed"} and cfg.noise_std > 0:
        stressed = stressed + torch.randn_like(stressed) * cfg.noise_std

    if cfg.scenario in {"burst", "mixed"} and cfg.burst_rate > 0:
        spike_mask = torch.rand(stressed.shape[0], stressed.shape[1], 1, device=stressed.device) < cfg.burst_rate
        amplitude = 2.0 + torch.rand_like(stressed) * 2.0
        stressed = stressed + spike_mask * amplitude

    if cfg.scenario in {"level_shift", "mixed"} and cfg.level_shift != 0:
        shift_start = stressed.shape[1] // 2
        stressed[:, shift_start:, :] = stressed[:, shift_start:, :] + cfg.level_shift

    return apply_imputation(stressed, cfg.imputation)
