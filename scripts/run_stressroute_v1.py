from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from race_forecast.config import CapacityConfig, StressConfig, TrainConfig, load_config
from race_forecast.data import build_window_bundle
from race_forecast.metrics import capacity_proxy, count_parameters, measure_latency, mse_mae
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


def raw_values(pred: np.ndarray, true: np.ndarray, target_mean: float, target_scale: float) -> tuple[np.ndarray, np.ndarray]:
    return pred * target_scale + target_mean, true * target_scale + target_mean


def score_prediction(
    pred: np.ndarray,
    true: np.ndarray,
    target_mean: float,
    target_scale: float,
    capacity_cfg: CapacityConfig,
) -> dict[str, float]:
    quality = mse_mae(pred, true)
    pred_raw, true_raw = raw_values(pred, true, target_mean, target_scale)
    cap = capacity_proxy(
        pred_raw,
        true_raw,
        headroom=capacity_cfg.headroom,
        under_cost=capacity_cfg.under_cost,
        over_cost=capacity_cfg.over_cost,
        demand_floor=capacity_cfg.demand_floor,
    )
    return {
        **quality,
        "capacity_under_rate": cap.under_rate,
        "capacity_over_rate": cap.over_rate,
        "capacity_mean_under": cap.mean_under,
        "capacity_mean_over": cap.mean_over,
        "capacity_cost": cap.cost,
    }


def telemetry_features(stress_name: str, stress_cfg: StressConfig) -> dict[str, float | str]:
    return {
        "stress": stress_name,
        "stress_family": stress_cfg.scenario,
        "missing_point_ratio": stress_cfg.missing_rate if stress_cfg.scenario == "missing_points" else 0.0,
        "missing_channel_ratio": stress_cfg.missing_rate if stress_cfg.scenario == "missing_variables" else 0.0,
        "delayed_tail_ratio": min(stress_cfg.delay_steps / 96.0, 1.0) if stress_cfg.scenario == "delayed_tail" else 0.0,
        "spike_score": stress_cfg.burst_rate if stress_cfg.scenario == "burst" else 0.0,
        "level_shift_score": abs(stress_cfg.level_shift) if stress_cfg.scenario == "level_shift" else 0.0,
    }


def choose_by_latency(latency: pd.DataFrame, budget_ms: float | None, fallback_model: str, routable_models: set[str]) -> set[str]:
    latency = latency[latency["model"].isin(routable_models)].copy()
    if budget_ms is None:
        return set(latency["model"].tolist())
    allowed = set(latency[latency["latency_p95_ms"] <= budget_ms]["model"].tolist())
    if allowed:
        return allowed
    return {fallback_model}


def route_model(
    stress_name: str,
    val_scores: pd.DataFrame,
    latency: pd.DataFrame,
    objective: str,
    budget_ms: float | None,
    routable_models: set[str],
) -> tuple[str, str]:
    allowed = choose_by_latency(latency, budget_ms, "dlinear", routable_models)
    subset = val_scores[(val_scores["stress"] == stress_name) & (val_scores["model"].isin(allowed))].copy()
    if subset.empty:
        subset = val_scores[val_scores["model"].isin(allowed)].copy()
    if subset.empty:
        return "dlinear", "fallback_no_allowed_model"

    if budget_ms is not None and budget_ms <= 0.2 and "dlinear" in allowed:
        return "dlinear", "strict_latency_rule"
    if stress_name == "missing_variables_30" and "patchtst" in allowed:
        return "patchtst", "metric_outage_rule"

    metric = "capacity_cost" if objective == "capacity" else "mse"
    winner = subset.sort_values([metric, "latency_p95_ms", "model"]).iloc[0]
    return str(winner["model"]), f"validation_best_{metric}"


def aggregate_oracle(predictions: dict[str, dict[str, np.ndarray]], true: np.ndarray, metric: str) -> tuple[np.ndarray, list[str]]:
    model_names = sorted(predictions)
    stacked = np.stack([predictions[name]["pred"] for name in model_names], axis=0)
    if metric == "mae":
        errors = np.mean(np.abs(stacked - true[None, :, :]), axis=2)
    else:
        errors = np.mean((stacked - true[None, :, :]) ** 2, axis=2)
    choices = np.argmin(errors, axis=0)
    selected = stacked[choices, np.arange(stacked.shape[1]), :]
    selected_names = [model_names[int(idx)] for idx in choices]
    return selected, selected_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StressRoute v1 on native AIOpsStressBench models.")
    parser.add_argument("--base-config", default="configs/alibaba2018_machine_usage.yaml")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--models", nargs="*", default=["last_value", "dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--routable-models", nargs="*", default=["dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--scenarios", nargs="*", default=["clean", "missing_30", "missing_variables_30", "delayed_12", "burst", "level_shift"])
    parser.add_argument("--latency-budgets", nargs="*", default=["0.2", "0.5", "1.0", "none"])
    parser.add_argument("--objectives", nargs="*", default=["capacity", "mse"])
    parser.add_argument("--output", default="outputs/stressroute_v1_summary.csv")
    parser.add_argument("--selection-output", default="outputs/stressroute_v1_selection.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    routable_models = set(args.routable_models)

    cfg = load_config(args.base_config)
    if args.device is not None:
        cfg.train.device = args.device
    set_seed(cfg.seed)
    device = select_device(cfg.train.device)
    bundle = build_window_bundle(cfg.data, cfg.seed)
    target_mean = float(bundle.scaler.mean_[cfg.data.target_metric])
    target_scale = float(bundle.scaler.scale_[cfg.data.target_metric])
    train_cfg = cfg.train
    val_loader = make_loader(bundle.val, train_cfg.batch_size, False, train_cfg.num_workers)
    test_loader = make_loader(bundle.test_clean, train_cfg.batch_size, False, train_cfg.num_workers)

    train_stress = SCENARIOS["missing_30"]
    models = {}
    latency_rows = []
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x[: min(32, sample_x.shape[0])]
    for model_name in args.models:
        set_seed(cfg.seed)
        model = train_model(
            name=model_name,
            train_dataset=bundle.train,
            val_dataset=bundle.val,
            train_cfg=train_cfg,
            stress_cfg=train_stress,
            device=device,
            input_len=cfg.data.input_len,
            pred_len=cfg.data.pred_len,
            n_metrics=len(bundle.metric_names),
            target_metric=cfg.data.target_metric,
        )
        models[model_name] = model
        latency = measure_latency(model, sample_x, train_cfg.latency_warmup, train_cfg.latency_iters, device)
        latency_rows.append({"model": model_name, "params": count_parameters(model), **latency})

    latency_frame = pd.DataFrame(latency_rows)
    val_rows = []
    test_rows = []
    selection_rows = []
    for scenario_name in args.scenarios:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        stress_cfg = SCENARIOS[scenario_name]
        val_predictions = {}
        test_predictions = {}
        true_test = None
        for model_name, model in models.items():
            val_pred, val_true = predict(model, val_loader, stress_cfg, device)
            test_pred, test_true = predict(model, test_loader, stress_cfg, device)
            val_score = score_prediction(val_pred, val_true, target_mean, target_scale, cfg.capacity)
            test_score = score_prediction(test_pred, test_true, target_mean, target_scale, cfg.capacity)
            latency_row = latency_frame[latency_frame["model"] == model_name].iloc[0].to_dict()
            val_rows.append({"source": args.source, "dataset": args.dataset, "stress": scenario_name, "model": model_name, **val_score, **latency_row})
            test_rows.append({"source": args.source, "dataset": args.dataset, "stress": scenario_name, "policy": "fixed", "model": model_name, **test_score, **latency_row})
            val_predictions[model_name] = {"pred": val_pred}
            test_predictions[model_name] = {"pred": test_pred}
            true_test = test_true

        if true_test is None:
            continue
        oracle_pred, oracle_choices = aggregate_oracle(test_predictions, true_test, "mse")
        oracle_score = score_prediction(oracle_pred, true_test, target_mean, target_scale, cfg.capacity)
        test_rows.append(
            {
                "source": args.source,
                "dataset": args.dataset,
                "stress": scenario_name,
                "policy": "oracle_mse",
                "model": "oracle_mse",
                **oracle_score,
                "latency_p50_ms": np.nan,
                "latency_p95_ms": np.nan,
                "latency_mean_ms": np.nan,
                "params": np.nan,
            }
        )
        selection_rows.extend(
            {
                "source": args.source,
                "dataset": args.dataset,
                "stress": scenario_name,
                "policy": "oracle_mse",
                "selected_model": choice,
            }
            for choice in oracle_choices
        )

    val_scores = pd.DataFrame(val_rows)
    for scenario_name in args.scenarios:
        stress_cfg = SCENARIOS[scenario_name]
        true_test = None
        test_predictions = {}
        for model_name, model in models.items():
            test_pred, test_true = predict(model, test_loader, stress_cfg, device)
            test_predictions[model_name] = {"pred": test_pred}
            true_test = test_true
        if true_test is None:
            continue

        for budget_text in args.latency_budgets:
            budget = None if budget_text.lower() == "none" else float(budget_text)
            for objective in args.objectives:
                selected, reason = route_model(scenario_name, val_scores, latency_frame, objective, budget, routable_models)
                pred = test_predictions[selected]["pred"]
                score = score_prediction(pred, true_test, target_mean, target_scale, cfg.capacity)
                latency_row = latency_frame[latency_frame["model"] == selected].iloc[0].to_dict()
                feature_row = telemetry_features(scenario_name, stress_cfg)
                test_rows.append(
                    {
                        "source": args.source,
                        "dataset": args.dataset,
                        "stress": scenario_name,
                        "policy": f"stressroute_v1_{objective}",
                        "model": selected,
                        "route_reason": reason,
                        "latency_budget_ms": np.nan if budget is None else budget,
                        **feature_row,
                        **score,
                        **latency_row,
                    }
                )
                selection_rows.append(
                    {
                        "source": args.source,
                        "dataset": args.dataset,
                        "stress": scenario_name,
                        "policy": f"stressroute_v1_{objective}",
                        "latency_budget_ms": np.nan if budget is None else budget,
                        "selected_model": selected,
                        "route_reason": reason,
                    }
                )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(test_rows)
    summary.to_csv(output, index=False)
    selection_output = Path(args.selection_output)
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(selection_output, index=False)
    metadata = {
        "base_config": args.base_config,
        "source": args.source,
        "dataset": args.dataset,
        "models": args.models,
        "scenarios": args.scenarios,
        "latency_budgets": args.latency_budgets,
        "objectives": args.objectives,
        "train_stress": asdict(train_stress),
    }
    (output.with_suffix(".metadata.json")).write_text(pd.Series(metadata).to_json(indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved {output}")
    print(f"Saved {selection_output}")


if __name__ == "__main__":
    main()
