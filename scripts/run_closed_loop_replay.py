from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset

from race_forecast.config import CapacityConfig, StressConfig, TrainConfig, load_config
from race_forecast.data import build_window_bundle
from race_forecast.metrics import measure_latency, mse_mae
from race_forecast.models import build_model, is_trainable
from race_forecast.stress import apply_stress
from race_forecast.train import make_loader, select_device


SCENARIOS = {
    "clean": StressConfig(scenario="clean"),
    "missing_30": StressConfig(scenario="missing_points", missing_rate=0.3),
    "missing_variables_30": StressConfig(scenario="missing_variables", missing_rate=0.3),
    "delayed_12": StressConfig(scenario="delayed_tail", delay_steps=12),
    "burst": StressConfig(scenario="burst", burst_rate=0.02),
    "level_shift": StressConfig(scenario="level_shift", level_shift=0.4),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(
    name: str,
    train_dataset,
    val_dataset,
    train_cfg: TrainConfig,
    stress_cfg: StressConfig,
    device: torch.device,
    input_len: int,
    pred_len: int,
    n_metrics: int,
    target_metric: int,
) -> nn.Module:
    model = build_model(name, input_len, pred_len, n_metrics, target_metric).to(device)
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
            if train_cfg.train_stress:
                x = apply_stress(x, stress_cfg, train=True)
            y = y.to(device)
            loss = criterion(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        val_pred, val_true = predict(model, val_loader, stress_cfg, device)
        val_loss = float(np.mean((val_pred - val_true) ** 2))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= train_cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def maybe_subset(dataset, max_windows: int | None):
    if max_windows is None or max_windows <= 0 or len(dataset) <= max_windows:
        return dataset
    return Subset(dataset, list(range(max_windows)))


@torch.no_grad()
def predict(model: nn.Module, loader, stress_cfg: StressConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    trues = []
    for x, y in loader:
        x = apply_stress(x.to(device), stress_cfg, train=False)
        preds.append(model(x).detach().cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


@torch.no_grad()
def collect_context(loader, stress_cfg: StressConfig, device: torch.device, target_metric: int) -> np.ndarray:
    contexts = []
    for x, _y in loader:
        x = apply_stress(x.to(device), stress_cfg, train=False).detach().cpu().numpy()
        contexts.append(x[:, :, target_metric])
    return np.concatenate(contexts, axis=0)


def to_raw(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return values * scale + mean


def forecast_capacity(pred_raw: np.ndarray, headroom: float) -> np.ndarray:
    return np.maximum(pred_raw, 0.0) * (1.0 + headroom)


def reactive_hpa_capacity(context_raw: np.ndarray, true_raw: np.ndarray, headroom: float) -> np.ndarray:
    previous = np.concatenate([context_raw[:, -1:], true_raw[:, :-1]], axis=1)
    return np.maximum(previous, 0.0) * (1.0 + headroom)


def oracle_capacity(predictions: dict[str, np.ndarray], true_raw: np.ndarray, headroom: float) -> tuple[np.ndarray, list[str]]:
    names = sorted(predictions)
    caps = np.stack([forecast_capacity(predictions[name], headroom) for name in names], axis=0)
    demand = np.maximum(true_raw, 0.0)
    under = np.maximum(demand[None, :, :] - caps, 0.0)
    over = np.maximum(caps - demand[None, :, :], 0.0)
    cost = np.mean(5.0 * under + over, axis=2)
    choice = np.argmin(cost, axis=0)
    selected = caps[choice, np.arange(caps.shape[1]), :]
    return selected, [names[int(idx)] for idx in choice]


def stressroute_choice(
    stress: str,
    objective: str,
    budget_ms: float | None,
    validation_scores: pd.DataFrame,
    latency_scores: pd.DataFrame,
    routable_models: list[str],
) -> str:
    allowed = set(routable_models)
    if budget_ms is not None:
        allowed = set(latency_scores[(latency_scores["model"].isin(routable_models)) & (latency_scores["latency_p95_ms"] <= budget_ms)]["model"])
        if not allowed:
            allowed = {"dlinear"}
    if budget_ms is not None and budget_ms <= 0.2 and "dlinear" in allowed:
        return "dlinear"
    metric = "capacity_cost" if objective == "capacity" else "mse"
    rows = validation_scores[(validation_scores["stress"] == stress) & (validation_scores["model"].isin(allowed))]
    if rows.empty:
        rows = validation_scores[validation_scores["model"].isin(allowed)]
    if rows.empty:
        return "dlinear"
    return str(rows.sort_values([metric, "latency_p95_ms", "model"]).iloc[0]["model"])


def control_metrics(
    capacity: np.ndarray,
    demand_raw: np.ndarray,
    target_utilization: float,
    min_replicas: int,
    max_replicas: int,
    max_scale_step: int,
    scale_up_cooldown: int,
    scale_down_cooldown: int,
    cold_start_delay: int,
    under_cost: float,
    over_cost: float,
    demand_floor: float,
    severe_threshold: float,
    normalization_quantile: float,
) -> dict[str, float]:
    demand = np.maximum(demand_raw, 0.0)
    cap = np.maximum(capacity, 0.0)
    unit = float(np.quantile(demand, normalization_quantile))
    unit = max(unit, float(demand_floor), 1e-6)
    demand = demand / unit
    cap = cap / unit
    desired = np.ceil(cap / max(target_utilization, 1e-6)).astype(int)
    desired = np.clip(desired, min_replicas, max_replicas)
    effective = np.zeros_like(desired, dtype=np.float32)
    pending: list[tuple[int, int]] = []
    current = max(min_replicas, int(desired[0, 0]))
    up_cd = 0
    down_cd = 0
    scale_actions = 0
    lag_steps = 0
    for i in range(desired.shape[0]):
        for t in range(desired.shape[1]):
            idx = i * desired.shape[1] + t
            ready = [replicas for ready_idx, replicas in pending if ready_idx <= idx]
            if ready:
                current = max(current, max(ready))
            pending = [(ready_idx, replicas) for ready_idx, replicas in pending if ready_idx > idx]
            want = int(desired[i, t])
            if want > current and up_cd <= 0:
                new_replicas = min(current + max_scale_step, want)
                pending.append((idx + cold_start_delay, new_replicas))
                scale_actions += 1
                up_cd = scale_up_cooldown
            elif want < current and down_cd <= 0:
                current = max(want, current - max_scale_step)
                scale_actions += 1
                down_cd = scale_down_cooldown
            up_cd = max(0, up_cd - 1)
            down_cd = max(0, down_cd - 1)
            effective[i, t] = current
            if current < want:
                lag_steps += 1

    provision = effective * target_utilization
    floor = max(float(demand_floor), 1e-6)
    denom = np.maximum(demand, floor)
    under = np.maximum(demand - provision, 0.0)
    over = np.maximum(provision - demand, 0.0)
    under_ratio = under / denom
    over_ratio = over / denom
    total_demand = float(np.sum(np.maximum(demand, floor)))
    under_area = float(np.sum(under) / total_demand)
    over_area = float(np.sum(over) / total_demand)
    return {
        "overload_duration": float(np.mean(under > 0.0)),
        "severe_overload_duration": float(np.mean(under_ratio > severe_threshold)),
        "peak_overload_ratio": float(np.max(under_ratio)),
        "replica_minutes": float(np.sum(effective)),
        "mean_replicas": float(np.mean(effective)),
        "over_provision_area": over_area,
        "scale_action_count": float(scale_actions),
        "scaling_lag_rate": float(lag_steps / desired.size),
        "total_control_cost": float(under_cost * under_area + over_cost * over_area),
    }


def score_arrays(pred_raw: np.ndarray, true_raw: np.ndarray, headroom: float, capacity_cfg: CapacityConfig) -> dict[str, float]:
    quality = mse_mae(pred_raw, true_raw)
    capacity = forecast_capacity(pred_raw, headroom)
    demand = np.maximum(true_raw, 0.0)
    provision = np.maximum(capacity, 0.0)
    floor = max(float(capacity_cfg.demand_floor), 1e-6)
    denom = np.maximum(demand, floor)
    under = np.maximum(demand - provision, 0.0)
    over = np.maximum(provision - demand, 0.0)
    cost = capacity_cfg.under_cost * (under / denom) + capacity_cfg.over_cost * (over / denom)
    return {**quality, "capacity_cost": float(np.mean(cost))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run closed-loop capacity replay probe without Docker/K8s.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", default="outputs/strong_probe")
    parser.add_argument("--models", nargs="*", default=["dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--scenarios", nargs="*", default=["clean", "missing_30", "missing_variables_30", "delayed_12", "burst", "level_shift"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-windows", type=int, default=0)
    parser.add_argument("--max-val-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latency-budgets", nargs="*", default=["0.2", "1.0", "3.0"])
    parser.add_argument("--target-utilization", type=float, default=0.7)
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=20)
    parser.add_argument("--max-scale-step", type=int, default=2)
    parser.add_argument("--scale-up-cooldown", type=int, default=1)
    parser.add_argument("--scale-down-cooldown", type=int, default=3)
    parser.add_argument("--cold-start-delay", type=int, default=2)
    parser.add_argument("--severe-threshold", type=float, default=0.1)
    parser.add_argument("--normalization-quantile", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_config(args.base_config)
    cfg.train.device = args.device
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    device = select_device(cfg.train.device)
    bundle = build_window_bundle(cfg.data, args.seed)
    target_mean = float(bundle.scaler.mean_[cfg.data.target_metric])
    target_scale = float(bundle.scaler.scale_[cfg.data.target_metric])
    train_cfg = cfg.train
    train_dataset = maybe_subset(bundle.train, args.max_train_windows)
    val_dataset = maybe_subset(bundle.val, args.max_val_windows)
    train_stress = SCENARIOS["missing_30"]
    train_loader_sample = make_loader(bundle.test_clean, train_cfg.batch_size, False, train_cfg.num_workers)
    sample_x, _ = next(iter(train_loader_sample))
    sample_x = sample_x[: min(32, sample_x.shape[0])]

    models: dict[str, nn.Module] = {}
    latency_rows = []
    for name in args.models:
        set_seed(args.seed)
        model = train_model(
            name,
            train_dataset,
            val_dataset,
            train_cfg,
            train_stress,
            device,
            cfg.data.input_len,
            cfg.data.pred_len,
            len(bundle.metric_names),
            cfg.data.target_metric,
        )
        models[name] = model
        latency = measure_latency(model, sample_x, train_cfg.latency_warmup, train_cfg.latency_iters, device)
        latency_rows.append({"model": name, **latency})
    latency_frame = pd.DataFrame(latency_rows)

    rows = []
    sensitivity_rows = []
    validation_scores = []
    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        stress_cfg = SCENARIOS[scenario]
        val_loader = make_loader(val_dataset, train_cfg.batch_size, False, train_cfg.num_workers)
        test_loader = make_loader(bundle.test_clean, train_cfg.batch_size, False, train_cfg.num_workers)
        context_scaled = collect_context(test_loader, stress_cfg, device, cfg.data.target_metric)[: args.max_test_windows]
        context_raw = to_raw(context_scaled, target_mean, target_scale)
        predictions: dict[str, np.ndarray] = {}
        true_raw = None
        for name, model in models.items():
            val_pred, val_true = predict(model, val_loader, stress_cfg, device)
            val_raw_pred = to_raw(val_pred, target_mean, target_scale)
            val_raw_true = to_raw(val_true, target_mean, target_scale)
            val_score = score_arrays(val_raw_pred, val_raw_true, cfg.capacity.headroom, cfg.capacity)
            validation_scores.append({"stress": scenario, "model": name, **val_score, **latency_frame[latency_frame["model"] == name].iloc[0].to_dict()})

            pred, true = predict(model, test_loader, stress_cfg, device)
            pred_raw = to_raw(pred[: args.max_test_windows], target_mean, target_scale)
            true_raw = to_raw(true[: args.max_test_windows], target_mean, target_scale)
            predictions[name] = pred_raw
        if true_raw is None:
            continue

        base_policies: dict[str, tuple[str, np.ndarray]] = {
            "reactive_hpa_like": ("reactive_hpa", reactive_hpa_capacity(context_raw, true_raw, cfg.capacity.headroom)),
        }
        for name, pred_raw in predictions.items():
            base_policies[f"forecast_{name}"] = (name, forecast_capacity(pred_raw, cfg.capacity.headroom))
        oracle_cap, oracle_choices = oracle_capacity(predictions, true_raw, cfg.capacity.headroom)
        base_policies["oracle_selector"] = ("oracle", oracle_cap)

        parsed_budgets = [None if str(text).lower() == "none" else float(text) for text in args.latency_budgets]
        for budget in parsed_budgets:
            policies = dict(base_policies)
            route_model = stressroute_choice(
                scenario,
                "capacity",
                budget,
                pd.DataFrame(validation_scores),
                latency_frame,
                args.models,
            )
            policies["stressroute_v1_capacity"] = (route_model, forecast_capacity(predictions[route_model], cfg.capacity.headroom))

            for policy, (model_name, capacity) in policies.items():
                pseudo_pred = capacity / (1.0 + cfg.capacity.headroom)
                metrics = control_metrics(
                    capacity,
                    true_raw,
                    args.target_utilization,
                    args.min_replicas,
                    args.max_replicas,
                    args.max_scale_step,
                    args.scale_up_cooldown,
                    args.scale_down_cooldown,
                    args.cold_start_delay,
                    cfg.capacity.under_cost,
                    cfg.capacity.over_cost,
                    cfg.capacity.demand_floor,
                    args.severe_threshold,
                    args.normalization_quantile,
                )
                quality = score_arrays(pseudo_pred, true_raw, cfg.capacity.headroom, cfg.capacity)
                latency = {"latency_p50_ms": 0.0, "latency_p95_ms": 0.0, "latency_mean_ms": 0.0}
                if model_name in predictions:
                    quality = score_arrays(predictions[model_name], true_raw, cfg.capacity.headroom, cfg.capacity)
                    latency = latency_frame[latency_frame["model"] == model_name].iloc[0].to_dict()
                rows.append(
                    {
                        "source": args.source,
                        "dataset": args.dataset,
                        "stress": scenario,
                        "policy": policy,
                        "model": model_name,
                        "seed": args.seed,
                        "latency_budget_ms": np.nan if budget is None else budget,
                        "target_utilization": args.target_utilization,
                        "headroom": cfg.capacity.headroom,
                        **quality,
                        **latency,
                        **metrics,
                    }
                )
        for cold_start in [0, 2, 4]:
            for cooldown in [1, 3]:
                sensitivity_budget_text = args.latency_budgets[-1]
                sensitivity_budget = None if str(sensitivity_budget_text).lower() == "none" else float(sensitivity_budget_text)
                route_model = stressroute_choice(
                    scenario,
                    "capacity",
                    sensitivity_budget,
                    pd.DataFrame(validation_scores),
                    latency_frame,
                    args.models,
                )
                sensitivity_policies = dict(base_policies)
                sensitivity_policies["stressroute_v1_capacity"] = (route_model, forecast_capacity(predictions[route_model], cfg.capacity.headroom))
                for key in ["forecast_dlinear", "forecast_patchtst", "stressroute_v1_capacity"]:
                    if key not in sensitivity_policies:
                        continue
                    metrics = control_metrics(
                        sensitivity_policies[key][1],
                        true_raw,
                        args.target_utilization,
                        args.min_replicas,
                        args.max_replicas,
                        args.max_scale_step,
                        args.scale_up_cooldown,
                        cooldown,
                        cold_start,
                        cfg.capacity.under_cost,
                        cfg.capacity.over_cost,
                        cfg.capacity.demand_floor,
                        args.severe_threshold,
                        args.normalization_quantile,
                    )
                    sensitivity_rows.append(
                        {
                            "source": args.source,
                            "dataset": args.dataset,
                            "stress": scenario,
                            "policy": key,
                            "cold_start_delay": cold_start,
                            "scale_down_cooldown": cooldown,
                            **metrics,
                        }
                    )

    summary = pd.DataFrame(rows)
    sensitivity = pd.DataFrame(sensitivity_rows)
    summary_path = out_root / f"closed_loop_replay_{args.dataset}_summary.csv"
    sensitivity_path = out_root / f"closed_loop_replay_{args.dataset}_sensitivity.csv"
    summary.to_csv(summary_path, index=False)
    sensitivity.to_csv(sensitivity_path, index=False)
    gate_path = out_root / f"closed_loop_decision_gate_{args.dataset}.md"
    gate_path.write_text(make_decision_gate(args.dataset, summary, sensitivity), encoding="utf-8")
    meta = {"args": vars(args), "config": args.base_config, "latency": latency_frame.to_dict(orient="records")}
    (out_root / f"closed_loop_replay_{args.dataset}_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved {summary_path}")
    print(f"Saved {sensitivity_path}")
    print(f"Saved {gate_path}")


def pct_change(new: float, base: float) -> float:
    if not np.isfinite(new) or not np.isfinite(base) or abs(base) < 1e-12:
        return float("nan")
    return float((new - base) / abs(base))


def make_decision_gate(dataset: str, summary: pd.DataFrame, sensitivity: pd.DataFrame) -> str:
    lines = [f"# Closed-loop Decision Gate: {dataset}", ""]
    strict_votes = 0
    relaxed_votes = 0
    budgets = sorted(float(v) for v in summary["latency_budget_ms"].dropna().unique())
    for budget in budgets:
      lines.append(f"## Latency budget {budget:g} ms")
      for stress in sorted(summary["stress"].dropna().unique()):
        sub = summary[(summary["stress"] == stress) & (summary["latency_budget_ms"] == budget)]
        dlinear = sub[sub["policy"] == "forecast_dlinear"]
        patch = sub[sub["policy"] == "forecast_patchtst"]
        route = sub[sub["policy"] == "stressroute_v1_capacity"]
        if dlinear.empty or patch.empty or route.empty:
            continue
        route_row = route.iloc[0]
        dlinear_row = dlinear.iloc[0]
        patch_row = patch.iloc[0]
        risk_vs_dlinear = pct_change(route_row["total_control_cost"], dlinear_row["total_control_cost"])
        replica_vs_patch = pct_change(route_row["replica_minutes"], patch_row["replica_minutes"])
        latency_vs_patch = pct_change(route_row.get("latency_p95_ms", np.nan), patch_row.get("latency_p95_ms", np.nan))
        if budget <= 0.2 and latency_vs_patch < -0.2:
            strict_votes += 1
        if budget > 0.2 and risk_vs_dlinear < -0.05:
            relaxed_votes += 1
        lines.append(
            f"- `{stress}`: StressRoute cost vs DLinear {risk_vs_dlinear:.2%}; "
            f"replica-minutes vs PatchTST {replica_vs_patch:.2%}; "
            f"P95 latency vs PatchTST {latency_vs_patch:.2%}."
        )
      lines.append("")
    stable = True
    if not sensitivity.empty:
        pivot = sensitivity.pivot_table(index=["stress", "cold_start_delay", "scale_down_cooldown"], columns="policy", values="total_control_cost")
        if {"forecast_dlinear", "forecast_patchtst", "stressroute_v1_capacity"}.issubset(pivot.columns):
            winners = pivot[["forecast_dlinear", "forecast_patchtst", "stressroute_v1_capacity"]].idxmin(axis=1)
            stable = "stressroute_v1_capacity" in set(winners)
            lines.append("")
            lines.append(f"Sensitivity winner includes StressRoute: `{stable}`.")
    verdict = "GO" if (strict_votes >= 1 and relaxed_votes >= 1 and stable) else "CONDITIONAL" if (strict_votes >= 1 or relaxed_votes >= 1) else "NO-GO"
    lines.insert(2, f"Verdict: **{verdict}**")
    lines.insert(3, "")
    if verdict == "GO":
        lines.append("")
        lines.append("Use this as a candidate strong-paper extension: closed-loop replay adds deployment-policy evidence beyond static capacity proxy.")
    else:
        lines.append("")
        lines.append("Keep the current 10-page paper as the main version; treat this replay as artifact-only unless later sandbox results improve.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
