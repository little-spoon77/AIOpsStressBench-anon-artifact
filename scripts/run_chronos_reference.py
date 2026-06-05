from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from race_forecast.config import StressConfig, load_config
from race_forecast.data import build_window_bundle
from race_forecast.metrics import capacity_proxy, mse_mae
from race_forecast.stress import apply_stress


SCENARIOS = {
    "clean": {"scenario": "clean"},
    "missing_30": {"scenario": "missing_points", "missing_rate": 0.3},
    "missing_variables_30": {"scenario": "missing_variables", "missing_rate": 0.3},
    "delayed_12": {"scenario": "delayed_tail", "delay_steps": 12},
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


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_pipeline(model_id: str, device: torch.device, dtype: str):
    from chronos import ChronosBoltPipeline, ChronosPipeline

    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    pipeline_cls = ChronosBoltPipeline if "bolt" in model_id.lower() else ChronosPipeline
    return pipeline_cls.from_pretrained(model_id, device_map=str(device), torch_dtype=torch_dtype)


def predict_median(pipeline, context: torch.Tensor, pred_len: int, sample_count: int) -> torch.Tensor:
    with torch.no_grad():
        forecast = pipeline.predict(context, prediction_length=pred_len)
    if forecast.ndim == 3:
        if forecast.shape[-1] == pred_len:
            return forecast.median(dim=1).values
        if forecast.shape[1] == pred_len:
            return forecast.median(dim=-1).values
    if forecast.ndim == 2:
        return forecast
    raise ValueError(f"Unexpected Chronos forecast shape: {tuple(forecast.shape)}")


def evaluate_chronos(
    pipeline,
    loader: DataLoader,
    stress_cfg: StressConfig,
    device: torch.device,
    pred_len: int,
    target_metric: int,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    trues = []
    for x, y in loader:
        x = apply_stress(x.to(device), stress_cfg, train=False)
        context = x[:, :, target_metric].contiguous()
        pred = predict_median(pipeline, context, pred_len=pred_len, sample_count=sample_count)
        preds.append(pred.detach().float().cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def measure_chronos_latency(
    pipeline,
    sample_x: torch.Tensor,
    stress_cfg: StressConfig,
    device: torch.device,
    pred_len: int,
    target_metric: int,
    warmup: int,
    iters: int,
    sample_count: int,
) -> dict[str, float]:
    x = apply_stress(sample_x.to(device), stress_cfg, train=False)[:, :, target_metric].contiguous()
    timings = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = predict_median(pipeline, x, pred_len=pred_len, sample_count=sample_count)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(iters):
            start = time.perf_counter()
            _ = predict_median(pipeline, x, pred_len=pred_len, sample_count=sample_count)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "latency_mean_ms": float(np.mean(timings)),
        "latency_p50_ms": float(np.percentile(timings, 50)),
        "latency_p95_ms": float(np.percentile(timings, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Chronos zero-shot reference inside AIOpsStressBench.")
    parser.add_argument("--model-id", default="amazon/chronos-bolt-tiny")
    parser.add_argument("--base-config", default="configs/alibaba2018_machine_usage.yaml")
    parser.add_argument("--summary", default="outputs/chronos_reference_summary.csv")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--scenarios", nargs="*", default=["clean", "missing_30", "missing_variables_30"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-iters", type=int, default=10)
    parser.add_argument("--sample-count", type=int, default=20)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.base_config)
    cfg.train.batch_size = args.batch_size
    cfg.train.device = args.device
    device = select_device(cfg.train.device)
    bundle = build_window_bundle(cfg.data, args.seed)
    if args.max_windows > 0:
        test_len = min(args.max_windows, len(bundle.test_clean))
        test_dataset = Subset(bundle.test_clean, list(range(test_len)))
    else:
        test_dataset = bundle.test_clean
    loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pipeline = load_pipeline(args.model_id, device=device, dtype=args.dtype)
    rows = []
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        stress_cfg = make_stress(SCENARIOS[scenario])
        pred, true = evaluate_chronos(
            pipeline=pipeline,
            loader=loader,
            stress_cfg=stress_cfg,
            device=device,
            pred_len=cfg.data.pred_len,
            target_metric=cfg.data.target_metric,
            sample_count=args.sample_count,
        )
        quality = mse_mae(pred, true)
        pred_raw = pred * float(bundle.scaler.scale_[cfg.data.target_metric]) + float(bundle.scaler.mean_[cfg.data.target_metric])
        true_raw = true * float(bundle.scaler.scale_[cfg.data.target_metric]) + float(bundle.scaler.mean_[cfg.data.target_metric])
        cap = capacity_proxy(
            pred_raw,
            true_raw,
            headroom=cfg.capacity.headroom,
            under_cost=cfg.capacity.under_cost,
            over_cost=cfg.capacity.over_cost,
            demand_floor=cfg.capacity.demand_floor,
        )
        sample_x, _ = next(iter(loader))
        sample_x = sample_x[: min(16, sample_x.shape[0])]
        latency = measure_chronos_latency(
            pipeline=pipeline,
            sample_x=sample_x,
            stress_cfg=stress_cfg,
            device=device,
            pred_len=cfg.data.pred_len,
            target_metric=cfg.data.target_metric,
            warmup=args.latency_warmup,
            iters=args.latency_iters,
            sample_count=args.sample_count,
        )
        row = {
            "source": args.source,
            "dataset": args.dataset,
            "stress": scenario,
            "model": "chronos_bolt_reference",
            "model_id": args.model_id,
            "zero_shot": 1,
            "eval_windows": len(test_dataset),
            "params": np.nan,
            **quality,
            **latency,
            "capacity_under_rate": cap.under_rate,
            "capacity_over_rate": cap.over_rate,
            "capacity_mean_under": cap.mean_under,
            "capacity_mean_over": cap.mean_over,
            "capacity_cost": cap.cost,
            "max_memory_mb": int(torch.cuda.max_memory_allocated(device) / 1024 / 1024) if device.type == "cuda" else 0,
        }
        rows.append(row)
        print(pd.DataFrame(rows).to_string(index=False))

    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
