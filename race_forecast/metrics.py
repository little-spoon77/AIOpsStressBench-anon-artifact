from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class CapacityResult:
    under_rate: float
    over_rate: float
    mean_under: float
    mean_over: float
    cost: float


def mse_mae(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = pred - true
    return {
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
    }


def capacity_proxy(
    pred: np.ndarray,
    true: np.ndarray,
    headroom: float,
    under_cost: float,
    over_cost: float,
    demand_floor: float,
) -> CapacityResult:
    provision = np.maximum(0.0, pred) * (1.0 + headroom)
    demand = np.maximum(0.0, true)
    under = np.maximum(0.0, demand - provision)
    over = np.maximum(0.0, provision - demand)
    floor = max(float(demand_floor), 1e-6)
    denom = np.maximum(np.abs(demand), floor)
    cost = under_cost * (under / denom) + over_cost * (over / denom)
    return CapacityResult(
        under_rate=float(np.mean(under > 0)),
        over_rate=float(np.mean(over > 0)),
        mean_under=float(np.mean(under)),
        mean_over=float(np.mean(over)),
        cost=float(np.mean(cost)),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


@torch.no_grad()
def measure_latency(
    model: nn.Module,
    sample: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    sample = sample.to(device)
    for _ in range(warmup):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    durations = []
    for _ in range(iters):
        start = time.perf_counter()
        _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(durations, dtype=np.float64)
    return {
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
        "latency_mean_ms": float(arr.mean()),
    }
