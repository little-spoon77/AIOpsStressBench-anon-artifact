from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_forecast.config import CapacityConfig, DataConfig, TrainConfig
from race_forecast.data import build_window_bundle, load_npz_cloudops, normalize_by_train
from race_forecast.metrics import capacity_proxy, measure_latency, mse_mae
from race_forecast.models import build_model, is_trainable
from race_forecast.stress import apply_stress
from race_forecast.train import make_loader, select_device
from race_forecast.config import StressConfig


DATASETS = {
    "alibaba2018": "data/alibaba2018_machine_usage.npz",
    "salesforce_borg": "data/salesforce_borg_256x2048.npz",
}

STRESSES = {
    "missing_30": StressConfig(scenario="missing_points", missing_rate=0.3),
    "missing_variables_30": StressConfig(scenario="missing_variables", missing_rate=0.3),
}

FORECASTERS = ["dlinear", "patchtst"]
IMPUTATIONS = ["none", "forward_fill", "mean", "learned_mask_imputer"]


class MaskedWindowDataset(Dataset):
    def __init__(self, base: Dataset, seed: int) -> None:
        self.base = base
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, _ = self.base[idx]
        clean = x.float()
        mode = self.rng.integers(0, 3)
        if mode == 0:
            mask = torch.rand_like(clean) < 0.1
        elif mode == 1:
            mask = torch.rand_like(clean) < 0.3
        else:
            channel_mask = torch.rand(1, clean.shape[-1]) < 0.3
            mask = channel_mask.expand_as(clean)
        degraded = clean.masked_fill(mask, 0.0)
        observed = (~mask).float()
        features = torch.cat([degraded, observed], dim=-1)
        return features, clean


class MaskAwareImputer(nn.Module):
    def __init__(self, n_metrics: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_metrics * 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_metrics),
        )

    def forward(self, degraded: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        proposal = self.net(torch.cat([degraded, observed], dim=-1))
        return torch.where(observed > 0.5, degraded, proposal)


def deterministic_subset_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit <= 0 or length <= limit:
        return list(range(length))
    return np.linspace(0, length - 1, num=limit, dtype=np.int64).tolist()


def target_stats(data_cfg: DataConfig) -> tuple[float, float]:
    raw, _, _ = load_npz_cloudops(data_cfg)
    train_end = int(raw.shape[1] * data_cfg.train_ratio)
    _, scaler = normalize_by_train(raw, train_end)
    target_mean = float(scaler.mean_[data_cfg.target_metric])
    target_scale = float(scaler.scale_[data_cfg.target_metric])
    return target_mean, target_scale


def train_imputer(
    train_dataset,
    val_dataset,
    n_metrics: int,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
) -> MaskAwareImputer:
    torch.manual_seed(seed)
    imputer = MaskAwareImputer(n_metrics=n_metrics).to(device)
    train_loader = DataLoader(MaskedWindowDataset(train_dataset, seed), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(MaskedWindowDataset(val_dataset, seed + 1), batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(imputer.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    for _ in range(epochs):
        imputer.train()
        for features, clean in train_loader:
            features = features.to(device)
            clean = clean.to(device)
            n = clean.shape[-1]
            pred = imputer(features[..., :n], features[..., n:])
            loss = criterion(pred, clean)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        imputer.eval()
        losses = []
        with torch.no_grad():
            for features, clean in val_loader:
                features = features.to(device)
                clean = clean.to(device)
                n = clean.shape[-1]
                pred = imputer(features[..., :n], features[..., n:])
                losses.append(float(criterion(pred, clean).detach().cpu()))
        val_loss = float(np.mean(losses)) if losses else float("inf")
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in imputer.state_dict().items()}
    if best_state is not None:
        imputer.load_state_dict(best_state)
    return imputer


def train_forecaster(name: str, bundle, data_cfg: DataConfig, train_cfg: TrainConfig, device: torch.device, train_dataset, val_dataset):
    torch.manual_seed(train_cfg.latency_iters + data_cfg.input_len)
    model = build_model(
        name,
        input_len=data_cfg.input_len,
        pred_len=data_cfg.pred_len,
        n_metrics=len(bundle.metric_names),
        target_metric=data_cfg.target_metric,
    ).to(device)
    if not is_trainable(model):
        return model
    train_loader = make_loader(train_dataset, train_cfg.batch_size, True, train_cfg.num_workers)
    val_loader = make_loader(val_dataset, train_cfg.batch_size, False, train_cfg.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    for _epoch in range(train_cfg.epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        vals = []
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device))
                vals.append(float(torch.mean((pred.cpu() - y) ** 2)))
        val_loss = float(np.mean(vals)) if vals else float("inf")
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
    return model


def simple_impute(x: torch.Tensor, method: str) -> torch.Tensor:
    if method == "none":
        return x
    if method == "forward_fill":
        observed = x.abs() > 1e-6
        out = x.clone()
        for t in range(1, x.shape[1]):
            out[:, t] = torch.where(observed[:, t], out[:, t], out[:, t - 1])
        return out
    if method == "mean":
        observed = x.abs() > 1e-6
        counts = observed.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        means = (x * observed.float()).sum(dim=1, keepdim=True) / counts
        return torch.where(observed, x, means)
    raise ValueError(method)


def predict_with_pipeline(model, imputer, loader, stress_cfg: StressConfig, imputation: str, device: torch.device):
    preds = []
    trues = []
    model.eval()
    if imputer is not None:
        imputer.eval()
    with torch.no_grad():
        for x, y in loader:
            stressed = apply_stress(x.to(device), replace_stress_imputation(stress_cfg, "none"), train=False)
            if imputation == "learned_mask_imputer":
                observed = (stressed.abs() > 1e-6).float()
                stressed = imputer(stressed, observed)
            else:
                stressed = simple_impute(stressed, imputation)
            pred = model(stressed)
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def replace_stress_imputation(cfg: StressConfig, imputation: str) -> StressConfig:
    return StressConfig(
        scenario=cfg.scenario,
        missing_rate=cfg.missing_rate,
        noise_std=cfg.noise_std,
        delay_steps=cfg.delay_steps,
        burst_rate=cfg.burst_rate,
        level_shift=cfg.level_shift,
        imputation=imputation,
    )


def score_prediction(pred, true, target_mean, target_scale, capacity_cfg):
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
    return {
        **quality,
        "capacity_cost": capacity.cost,
        "capacity_under_rate": capacity.under_rate,
        "capacity_over_rate": capacity.over_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded learned imputation + forecasting pipeline baselines.")
    parser.add_argument("--output-dir", default="outputs/learned_imputation_pipeline")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--imputer-epochs", type=int, default=5)
    parser.add_argument("--forecaster-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-windows", type=int, default=8192)
    parser.add_argument("--max-val-windows", type=int, default=2048)
    parser.add_argument("--max-test-windows", type=int, default=12000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    capacity_cfg = CapacityConfig()
    rows = []

    for dataset_name, npz_path in DATASETS.items():
        data_cfg = DataConfig(source="npz", npz_path=npz_path, input_len=96, pred_len=24, target_metric=0)
        bundle = build_window_bundle(data_cfg, args.seed)
        target_mean, target_scale = target_stats(data_cfg)
        train_idx = deterministic_subset_indices(len(bundle.train), args.max_train_windows)
        val_idx = deterministic_subset_indices(len(bundle.val), args.max_val_windows)
        test_idx = deterministic_subset_indices(len(bundle.test_clean), args.max_test_windows)
        train_dataset = Subset(bundle.train, train_idx)
        val_dataset = Subset(bundle.val, val_idx)
        test_dataset = Subset(bundle.test_clean, test_idx)
        train_cfg = TrainConfig(
            models=FORECASTERS,
            epochs=args.forecaster_epochs,
            batch_size=args.batch_size,
            patience=2,
            device=args.device,
            train_stress=False,
            calibrate=False,
        )
        imputer = train_imputer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            n_metrics=len(bundle.metric_names),
            device=device,
            seed=args.seed,
            epochs=args.imputer_epochs,
            batch_size=args.batch_size,
        )
        forecasters = {
            name: train_forecaster(name, bundle, data_cfg, train_cfg, device, train_dataset, val_dataset)
            for name in FORECASTERS
        }
        loader = make_loader(test_dataset, train_cfg.batch_size, False, train_cfg.num_workers)
        sample_x, _ = next(iter(loader))
        sample_x = sample_x[: min(32, sample_x.shape[0])]
        for stress_name, stress_cfg in STRESSES.items():
            for forecaster_name, model in forecasters.items():
                for imputation in IMPUTATIONS:
                    pred, true = predict_with_pipeline(
                        model=model,
                        imputer=imputer if imputation == "learned_mask_imputer" else None,
                        loader=loader,
                        stress_cfg=stress_cfg,
                        imputation=imputation,
                        device=device,
                    )
                    score = score_prediction(pred, true, target_mean, target_scale, capacity_cfg)
                    stressed_sample = apply_stress(sample_x.to(device), replace_stress_imputation(stress_cfg, "none"), train=False)
                    if imputation == "learned_mask_imputer":
                        observed = (stressed_sample.abs() > 1e-6).float()
                        stressed_sample = imputer(stressed_sample, observed)
                    else:
                        stressed_sample = simple_impute(stressed_sample, imputation)
                    latency = measure_latency(
                        model,
                        stressed_sample.detach().cpu(),
                        warmup=train_cfg.latency_warmup,
                        iters=train_cfg.latency_iters,
                        device=device,
                    )
                    rows.append(
                        {
                            "dataset": dataset_name,
                            "stress": stress_name,
                            "forecaster": forecaster_name,
                            "imputation": imputation,
                            "eval_windows": len(test_dataset),
                            **score,
                            **latency,
                        }
                    )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "learned_imputation_metrics.csv", index=False)

    summary_rows = []
    for (dataset, stress, forecaster), group in metrics.groupby(["dataset", "stress", "forecaster"]):
        best = group.sort_values(["capacity_cost", "mse", "imputation"]).iloc[0]
        baseline = group[group["imputation"] == "none"].iloc[0]
        learned = group[group["imputation"] == "learned_mask_imputer"].iloc[0]
        summary_rows.append(
            {
                "dataset": dataset,
                "stress": stress,
                "forecaster": forecaster,
                "best_imputation_by_capacity": best["imputation"],
                "learned_vs_none_capacity_pct": (learned["capacity_cost"] / baseline["capacity_cost"] - 1.0) * 100.0,
                "learned_vs_none_mse_pct": (learned["mse"] / baseline["mse"] - 1.0) * 100.0,
                "learned_capacity_cost": learned["capacity_cost"],
                "none_capacity_cost": baseline["capacity_cost"],
                "learned_mse": learned["mse"],
                "none_mse": baseline["mse"],
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "learned_imputation_summary.csv", index=False)

    channel = summary[summary["stress"] == "missing_variables_30"]
    point = summary[summary["stress"] == "missing_30"]
    learned_helped_any = bool((summary["learned_vs_none_capacity_pct"] < -1.0).any())
    channel_prefers_learned = bool((channel["best_imputation_by_capacity"] == "learned_mask_imputer").all()) if len(channel) else False
    decision = [
        "# Learned Imputation Pipeline Decision",
        "",
        "- Scope: bounded learned mask-aware imputation baseline, not SAITS/BRITS/PyPOTS.",
        "- Data: Alibaba 2018 and Salesforce/Borg.",
        "- Stress: missing_30 and missing_variables_30.",
        "- Forecasters: DLinear and PatchTST-lite.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Decision",
        "",
        f"- Learned imputation improves at least one pipeline by capacity: {'yes' if learned_helped_any else 'no'}.",
        f"- Channel-outage rows prefer learned imputation by capacity winner: {'yes' if channel_prefers_learned else 'no'}.",
        "- Recommended paper use: artifact-first; add a short imputation-scope sentence only if the result is stable and space allows.",
        "",
        "## Interpretation",
        "",
        "- This is a bounded learned imputation baseline, not a complete missing-data model benchmark.",
        "- This result should not be described as solving telemetry outage; it indicates that preprocessing can change the deployment pipeline choice.",
        "- If learned imputation is the best pipeline in several rows, the benchmark can report preprocessing as a deployment choice.",
    ]
    (output_dir / "learned_imputation_decision.md").write_text("\n".join(decision), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "rows": len(metrics), "learned_helped_any": learned_helped_any}, indent=2))


if __name__ == "__main__":
    main()
