from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from race_forecast.config import StressConfig, load_config
from race_forecast.data import build_window_bundle
from race_forecast.metrics import capacity_proxy, count_parameters, measure_latency, mse_mae
from race_forecast.models import build_model, is_trainable
from race_forecast.stress import apply_stress
from race_forecast.train import _calibrate_if_needed, _predict, make_loader, select_device


SCENARIOS = {
    "clean": StressConfig(scenario="clean"),
    "missing_30": StressConfig(scenario="missing_points", missing_rate=0.3),
    "missing_variables_30": StressConfig(scenario="missing_variables", missing_rate=0.3),
    "delayed_12": StressConfig(scenario="delayed_tail", delay_steps=12),
    "burst": StressConfig(scenario="burst", burst_rate=0.02),
    "level_shift": StressConfig(scenario="level_shift", level_shift=0.4),
}


POLICY_ORDER = {
    "fixed_dlinear": 0,
    "fixed_patchtst": 1,
    "stressroute_v1": 2,
    "stressroute_v2": 3,
    "oracle": 4,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_model(name, train_dataset, val_dataset, train_cfg, stress_cfg, device, input_len, pred_len, n_metrics, target_metric):
    model = build_model(name, input_len, pred_len, n_metrics, target_metric).to(device)
    if not is_trainable(model):
        return model
    train_loader = make_loader(train_dataset, train_cfg.batch_size, True, train_cfg.num_workers)
    val_loader = make_loader(val_dataset, train_cfg.batch_size, False, train_cfg.num_workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    criterion = torch.nn.MSELoss()
    best_state = None
    best_val = float("inf")
    stale = 0
    for _ in range(train_cfg.epochs):
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
        val_pred, val_true = _predict(model, val_loader, stress_cfg, device)
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


def raw_values(values: np.ndarray, mean: float, scale: float) -> np.ndarray:
    return values * scale + mean


def window_features(x: np.ndarray, stress_name: str, stress_cfg: StressConfig, latency_budget: float | None) -> np.ndarray:
    target = x[:, :, 0]
    diff = np.diff(target, axis=1)
    abs_mean = np.mean(np.abs(target), axis=1)
    volatility = np.std(diff, axis=1)
    trend = target[:, -1] - target[:, 0]
    zero_ratio = np.mean(np.isclose(x, 0.0), axis=(1, 2))
    tail_zero_ratio = np.mean(np.isclose(x[:, -max(1, min(stress_cfg.delay_steps or 1, x.shape[1])) :, :], 0.0), axis=(1, 2))
    feature = np.column_stack(
        [
            np.full(x.shape[0], stress_cfg.missing_rate if stress_cfg.scenario == "missing_points" else 0.0),
            np.full(x.shape[0], stress_cfg.missing_rate if stress_cfg.scenario == "missing_variables" else 0.0),
            np.full(x.shape[0], min(stress_cfg.delay_steps / 96.0, 1.0) if stress_cfg.scenario == "delayed_tail" else 0.0),
            np.full(x.shape[0], stress_cfg.noise_std if stress_cfg.scenario == "noisy" else 0.0),
            np.full(x.shape[0], stress_cfg.burst_rate if stress_cfg.scenario == "burst" else 0.0),
            np.full(x.shape[0], abs(stress_cfg.level_shift) if stress_cfg.scenario == "level_shift" else 0.0),
            np.full(x.shape[0], 1.0 if stress_name == "clean" else 0.0),
            zero_ratio,
            tail_zero_ratio,
            abs_mean,
            volatility,
            trend,
            np.full(x.shape[0], -1.0 if latency_budget is None else latency_budget),
        ]
    )
    return np.nan_to_num(feature.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def collect_features(loader, stress_name: str, stress_cfg: StressConfig, device: torch.device, latency_budget: float | None) -> np.ndarray:
    features = []
    with torch.no_grad():
        for x, _ in loader:
            stressed = apply_stress(x.to(device), stress_cfg, train=False).detach().cpu().numpy()
            features.append(window_features(stressed, stress_name, stress_cfg, latency_budget))
    return np.concatenate(features, axis=0)


def score_arrays(pred: np.ndarray, true: np.ndarray, target_mean: float, target_scale: float, capacity_cfg) -> dict[str, float]:
    quality = mse_mae(pred, true)
    pred_raw = raw_values(pred, target_mean, target_scale)
    true_raw = raw_values(true, target_mean, target_scale)
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
        "capacity_cost": cap.cost,
    }


def per_window_metric(pred: np.ndarray, true: np.ndarray, target_mean: float, target_scale: float, capacity_cfg, objective: str) -> np.ndarray:
    if objective == "mse":
        return np.mean((pred - true) ** 2, axis=1)
    pred_raw = raw_values(pred, target_mean, target_scale)
    true_raw = raw_values(true, target_mean, target_scale)
    provision = np.maximum(pred_raw, 0.0) * (1.0 + capacity_cfg.headroom)
    demand = np.maximum(true_raw, 0.0)
    denom = np.maximum(demand, max(float(capacity_cfg.demand_floor), 1e-6))
    under = np.maximum(demand - provision, 0.0)
    over = np.maximum(provision - demand, 0.0)
    cost = capacity_cfg.under_cost * (under / denom) + capacity_cfg.over_cost * (over / denom)
    return np.mean(cost, axis=1)


def select_by_names(predictions: dict[str, np.ndarray], selected_names: list[str]) -> np.ndarray:
    first = next(iter(predictions.values()))
    selected = np.zeros_like(first)
    for model_name in sorted(predictions):
        mask = np.array([name == model_name for name in selected_names])
        if np.any(mask):
            selected[mask] = predictions[model_name][mask]
    return selected


def best_names(predictions: dict[str, np.ndarray], true: np.ndarray, target_mean: float, target_scale: float, capacity_cfg, objective: str, allowed: set[str]) -> list[str]:
    names = sorted(allowed)
    scores = np.stack(
        [per_window_metric(predictions[name], true, target_mean, target_scale, capacity_cfg, objective) for name in names],
        axis=0,
    )
    choices = np.argmin(scores, axis=0)
    return [names[int(idx)] for idx in choices]


def latency_allowed(latency_frame: pd.DataFrame, budget: float | None, routable: list[str]) -> set[str]:
    if budget is None:
        return set(routable)
    allowed = set(latency_frame[(latency_frame["model"].isin(routable)) & (latency_frame["latency_p95_ms"] <= budget)]["model"])
    return allowed or {"dlinear"}


def choose_v1(stress_name: str, val_scores: pd.DataFrame, latency_frame: pd.DataFrame, objective: str, budget: float | None, routable: list[str]) -> str:
    allowed = latency_allowed(latency_frame, budget, routable)
    if budget is not None and budget <= 0.2 and "dlinear" in allowed:
        return "dlinear"
    if stress_name == "missing_variables_30" and "patchtst" in allowed:
        return "patchtst"
    metric = "capacity_cost" if objective == "capacity" else "mse"
    subset = val_scores[(val_scores["stress"] == stress_name) & (val_scores["model"].isin(allowed))].copy()
    if subset.empty:
        subset = val_scores[val_scores["model"].isin(allowed)].copy()
    if subset.empty:
        return "dlinear"
    return str(subset.sort_values([metric, "latency_p95_ms", "model"]).iloc[0]["model"])


def fit_router(features: np.ndarray, labels: list[str], default_model: str):
    unique = sorted(set(labels))
    if len(unique) < 2:
        return None, default_model
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    clf.fit(features, labels)
    return clf, default_model


def predict_router(clf, default_model: str, features: np.ndarray, allowed: set[str]) -> list[str]:
    if clf is None:
        return [default_model if default_model in allowed else sorted(allowed)[0]] * features.shape[0]
    predictions = clf.predict(features).tolist()
    fallback = default_model if default_model in allowed else sorted(allowed)[0]
    return [name if name in allowed else fallback for name in predictions]


def latency_stats(
    latency_frame: pd.DataFrame,
    selected_model: str,
    selected_names: list[str] | None = None,
) -> tuple[float, float, int]:
    if selected_names is None:
        selected_names = [selected_model]
    model_names = sorted({name for name in selected_names if name in set(latency_frame["model"])})
    if not model_names:
        return np.nan, np.nan, 0
    rows = latency_frame[latency_frame["model"].isin(model_names)]
    return float(rows["latency_p95_ms"].max()), float(rows["params"].max()), len(model_names)


def row_for_policy(
    source,
    dataset,
    stress_name,
    budget,
    objective,
    policy,
    selected_model,
    pred,
    true,
    target_mean,
    target_scale,
    capacity_cfg,
    latency_frame,
    dlinear_score,
    patchtst_score,
    oracle_score,
    selected_names: list[str] | None = None,
):
    score = score_arrays(pred, true, target_mean, target_scale, capacity_cfg)
    p95, params, route_model_count = latency_stats(latency_frame, selected_model, selected_names)
    budget_feasible = bool(pd.isna(p95) or budget is None or p95 <= budget)
    patchtst_latency = patchtst_score.get("latency_p95_ms", np.nan)
    if objective == "mse":
        regret = score["mse"] - oracle_score["mse"]
        oracle_denom = oracle_score["mse"]
    else:
        regret = score["capacity_cost"] - oracle_score["capacity_cost"]
        oracle_denom = oracle_score["capacity_cost"]
    return {
        "source": source,
        "dataset": dataset,
        "stress": stress_name,
        "latency_budget_ms": np.nan if budget is None else budget,
        "objective": objective,
        "policy": policy,
        "selected_model": selected_model,
        **score,
        "latency_p95_ms": p95,
        "params": params,
        "budget_feasible": budget_feasible,
        "route_model_count": route_model_count,
        "mse_vs_dlinear": (score["mse"] - dlinear_score["mse"]) / dlinear_score["mse"] if dlinear_score["mse"] else np.nan,
        "capacity_cost_vs_dlinear": (score["capacity_cost"] - dlinear_score["capacity_cost"]) / dlinear_score["capacity_cost"] if dlinear_score["capacity_cost"] else np.nan,
        "latency_vs_dlinear": (p95 - dlinear_score["latency_p95_ms"]) / dlinear_score["latency_p95_ms"] if dlinear_score["latency_p95_ms"] and not pd.isna(p95) else np.nan,
        "capacity_cost_vs_patchtst": (score["capacity_cost"] - patchtst_score["capacity_cost"]) / patchtst_score["capacity_cost"] if patchtst_score.get("capacity_cost") else np.nan,
        "latency_vs_patchtst": (p95 - patchtst_latency) / patchtst_latency if patchtst_latency and not pd.isna(p95) else np.nan,
        "latency_constrained_regret": regret,
        "latency_constrained_regret_ratio": regret / oracle_denom if oracle_denom else np.nan,
        "mse_oracle_gap": (score["mse"] - oracle_score["mse"]) / oracle_score["mse"] if oracle_score["mse"] else np.nan,
        "capacity_oracle_gap": (score["capacity_cost"] - oracle_score["capacity_cost"]) / oracle_score["capacity_cost"] if oracle_score["capacity_cost"] else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StressRoute v2 lightweight routing policy.")
    parser.add_argument("--base-config", default="configs/alibaba2018_machine_usage.yaml")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--models", nargs="*", default=["dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--routable-models", nargs="*", default=["dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--scenarios", nargs="*", default=["clean", "missing_30", "missing_variables_30", "delayed_12"])
    parser.add_argument("--latency-budgets", nargs="*", default=["0.2", "1.0", "none"])
    parser.add_argument("--objectives", nargs="*", default=["capacity", "mse"])
    parser.add_argument("--output", default="outputs/stressroute_v2_summary.csv")
    parser.add_argument("--selection-output", default="outputs/stressroute_v2_selection.csv")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

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

    models = {}
    latency_rows = []
    sample_x, _ = next(iter(test_loader))
    sample_x = sample_x[: min(32, sample_x.shape[0])]
    train_stress = SCENARIOS["missing_30"]
    for model_name in args.models:
        set_seed(cfg.seed)
        model = train_model(
            model_name,
            bundle.train,
            bundle.val,
            train_cfg,
            train_stress,
            device,
            cfg.data.input_len,
            cfg.data.pred_len,
            len(bundle.metric_names),
            cfg.data.target_metric,
        )
        models[model_name] = model
        latency = measure_latency(model, sample_x, train_cfg.latency_warmup, train_cfg.latency_iters, device)
        latency_rows.append({"model": model_name, "params": count_parameters(model), **latency})
    latency_frame = pd.DataFrame(latency_rows)

    predictions = {}
    val_scores_rows = []
    for scenario_name in args.scenarios:
        stress_cfg = SCENARIOS[scenario_name]
        predictions[scenario_name] = {"val": {}, "test": {}}
        for model_name, model in models.items():
            val_pred, val_true = _predict(model, val_loader, stress_cfg, device)
            val_pred, _ = _calibrate_if_needed(model, model_name, val_loader, val_pred, stress_cfg, device, train_cfg.calibrate)
            test_pred, test_true = _predict(model, test_loader, stress_cfg, device)
            test_pred, _ = _calibrate_if_needed(model, model_name, val_loader, test_pred, stress_cfg, device, train_cfg.calibrate)
            predictions[scenario_name]["val"][model_name] = val_pred
            predictions[scenario_name]["test"][model_name] = test_pred
            predictions[scenario_name]["val_true"] = val_true
            predictions[scenario_name]["test_true"] = test_true
            score = score_arrays(val_pred, val_true, target_mean, target_scale, cfg.capacity)
            latency_row = latency_frame[latency_frame["model"] == model_name].iloc[0].to_dict()
            val_scores_rows.append({"stress": scenario_name, "model": model_name, **score, **latency_row})
    val_scores = pd.DataFrame(val_scores_rows)

    rows = []
    selection_rows = []
    for budget_text in args.latency_budgets:
        budget = None if budget_text.lower() == "none" else float(budget_text)
        allowed = latency_allowed(latency_frame, budget, args.routable_models)
        for objective in args.objectives:
            feature_rows = []
            label_rows = []
            for scenario_name in args.scenarios:
                stress_cfg = SCENARIOS[scenario_name]
                features = collect_features(val_loader, scenario_name, stress_cfg, device, budget)
                labels = best_names(
                    predictions[scenario_name]["val"],
                    predictions[scenario_name]["val_true"],
                    target_mean,
                    target_scale,
                    cfg.capacity,
                    objective,
                    allowed,
                )
                feature_rows.append(features)
                label_rows.extend(labels)
            train_features = np.concatenate(feature_rows, axis=0)
            default_model = choose_v1(args.scenarios[0], val_scores, latency_frame, objective, budget, args.routable_models)
            clf, default_model = fit_router(train_features, label_rows, default_model)

            for scenario_name in args.scenarios:
                stress_cfg = SCENARIOS[scenario_name]
                true = predictions[scenario_name]["test_true"]
                test_predictions = predictions[scenario_name]["test"]
                dlinear_pred = test_predictions["dlinear"]
                dlinear_score = score_arrays(dlinear_pred, true, target_mean, target_scale, cfg.capacity)
                dlinear_latency = latency_frame[latency_frame["model"] == "dlinear"].iloc[0]
                dlinear_score["latency_p95_ms"] = float(dlinear_latency["latency_p95_ms"])
                if "patchtst" in test_predictions:
                    patchtst_score = score_arrays(test_predictions["patchtst"], true, target_mean, target_scale, cfg.capacity)
                    patchtst_latency = latency_frame[latency_frame["model"] == "patchtst"].iloc[0]
                    patchtst_score["latency_p95_ms"] = float(patchtst_latency["latency_p95_ms"])
                else:
                    patchtst_score = {"mse": np.nan, "capacity_cost": np.nan, "latency_p95_ms": np.nan}

                oracle_names = best_names(test_predictions, true, target_mean, target_scale, cfg.capacity, objective, allowed)
                oracle_pred = select_by_names(test_predictions, oracle_names)
                oracle_score = score_arrays(oracle_pred, true, target_mean, target_scale, cfg.capacity)

                fixed_policies = [("fixed_dlinear", "dlinear")]
                if "patchtst" in test_predictions:
                    fixed_policies.append(("fixed_patchtst", "patchtst"))
                for policy, model_name in fixed_policies:
                    pred = test_predictions[model_name]
                    rows.append(
                        row_for_policy(
                            args.source,
                            args.dataset,
                            scenario_name,
                            budget,
                            objective,
                            policy,
                            model_name,
                            pred,
                            true,
                            target_mean,
                            target_scale,
                            cfg.capacity,
                            latency_frame,
                            dlinear_score,
                            patchtst_score,
                            oracle_score,
                            selected_names=[model_name],
                        )
                    )

                v1_model = choose_v1(scenario_name, val_scores, latency_frame, objective, budget, args.routable_models)
                v1_pred = test_predictions[v1_model]
                rows.append(
                    row_for_policy(
                        args.source,
                        args.dataset,
                        scenario_name,
                        budget,
                        objective,
                        "stressroute_v1",
                        v1_model,
                        v1_pred,
                        true,
                        target_mean,
                        target_scale,
                        cfg.capacity,
                        latency_frame,
                        dlinear_score,
                        patchtst_score,
                        oracle_score,
                        selected_names=[v1_model],
                    )
                )

                test_features = collect_features(test_loader, scenario_name, stress_cfg, device, budget)
                v2_names = predict_router(clf, default_model, test_features, allowed)
                v2_pred = select_by_names(test_predictions, v2_names)
                majority_model = Counter(v2_names).most_common(1)[0][0]
                rows.append(
                    row_for_policy(
                        args.source,
                        args.dataset,
                        scenario_name,
                        budget,
                        objective,
                        "stressroute_v2",
                        majority_model,
                        v2_pred,
                        true,
                        target_mean,
                        target_scale,
                        cfg.capacity,
                        latency_frame,
                        dlinear_score,
                        patchtst_score,
                        oracle_score,
                        selected_names=v2_names,
                    )
                )

                rows.append(
                    row_for_policy(
                        args.source,
                        args.dataset,
                        scenario_name,
                        budget,
                        objective,
                        "oracle",
                        "oracle",
                        oracle_pred,
                        true,
                        target_mean,
                        target_scale,
                        cfg.capacity,
                        latency_frame,
                        dlinear_score,
                        patchtst_score,
                        oracle_score,
                        selected_names=oracle_names,
                    )
                )

                for policy, names in [("stressroute_v2", v2_names), ("oracle", oracle_names)]:
                    counts = Counter(names)
                    total = max(1, sum(counts.values()))
                    for model_name, count in counts.items():
                        selection_rows.append(
                            {
                                "source": args.source,
                                "dataset": args.dataset,
                                "stress": scenario_name,
                                "latency_budget_ms": np.nan if budget is None else budget,
                                "objective": objective,
                                "policy": policy,
                                "selected_model": model_name,
                                "count": count,
                                "share": count / total,
                            }
                        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary["policy_order"] = summary["policy"].map(POLICY_ORDER).fillna(99)
    summary = summary.sort_values(["source", "stress", "objective", "latency_budget_ms", "policy_order"])
    summary.to_csv(output, index=False)

    selection_output = Path(args.selection_output)
    selection_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(selection_output, index=False)
    print(summary.to_string(index=False))
    print(f"Saved {output}")
    print(f"Saved {selection_output}")


if __name__ == "__main__":
    main()
