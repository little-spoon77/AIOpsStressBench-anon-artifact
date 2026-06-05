from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_forecast.config import CapacityConfig, DataConfig, StressConfig, TrainConfig
from race_forecast.data import build_window_bundle, load_npz_cloudops, normalize_by_train
from race_forecast.metrics import capacity_proxy, count_parameters, measure_latency, mse_mae
from race_forecast.models import build_model, is_trainable
from race_forecast.train import make_loader, select_device


DATASETS = {
    "alibaba2018": "data/alibaba2018_machine_usage.npz",
    "salesforce_borg": "data/salesforce_borg_256x2048.npz",
}

MODELS = ["dlinear", "race_dlinear", "patchtst"]
PROXY_SLICES = [
    "high_flatline_or_zero_channel",
    "high_spike_score",
    "high_level_shift_score",
    "tail_flatline_proxy",
]

PROXY_LABELS = {
    "high_flatline_or_zero_channel": "flatline_or_zero_run",
    "high_spike_score": "spike_z6",
    "high_level_shift_score": "level_shift_top5",
    "tail_flatline_proxy": "tail_flatline12",
}


def robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median))
    scale = 1.4826 * mad if mad > 1e-12 else np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - median) / scale).astype(np.float32)


def level_shift_score(window: np.ndarray) -> float:
    half = window.shape[0] // 2
    if half == 0:
        return 0.0
    left = window[:half]
    right = window[half:]
    spread = np.nanstd(window)
    if not np.isfinite(spread) or spread < 1e-6:
        return 0.0
    return float(abs(np.nanmean(right) - np.nanmean(left)) / spread)


def window_proxy_scores(x: np.ndarray, metric_mask: np.ndarray | None = None) -> dict[str, float]:
    """Compute natural-degradation proxy scores for one normalized input window."""
    if metric_mask is not None:
        x = x[:, metric_mask]
    if x.shape[1] == 0:
        return {name: 0.0 for name in PROXY_SLICES}
    finite = np.isfinite(x)
    values = np.where(finite, x, np.nan)
    zero_rate_by_channel = np.nanmean(np.abs(values) <= 1e-6, axis=0)
    diffs = np.abs(np.diff(values, axis=0))
    flat_rate_by_channel = np.nanmean(diffs <= 1e-6, axis=0)
    flat_or_zero = float(np.nanmax(np.maximum(zero_rate_by_channel, flat_rate_by_channel)))

    spike_scores = []
    shift_scores = []
    tail_flat = []
    for metric_idx in range(values.shape[1]):
        channel = values[:, metric_idx]
        if np.all(~np.isfinite(channel)):
            spike_scores.append(0.0)
            shift_scores.append(0.0)
            tail_flat.append(0.0)
            continue
        z = robust_z(channel)
        spike_scores.append(float(np.nanmean(np.abs(z) > 6.0)))
        shift_scores.append(level_shift_score(channel))
        if len(channel) >= 13:
            tail = channel[-12:]
            tail_flat.append(float(np.nanmax(tail) - np.nanmin(tail) <= 1e-6))
        else:
            tail_flat.append(0.0)

    return {
        "high_flatline_or_zero_channel": flat_or_zero,
        "high_spike_score": float(np.nanmax(spike_scores)) if spike_scores else 0.0,
        "high_level_shift_score": float(np.nanmax(shift_scores)) if shift_scores else 0.0,
        "tail_flatline_proxy": float(np.nanmax(tail_flat)) if tail_flat else 0.0,
    }


def deterministic_subset_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit <= 0 or length <= limit:
        return list(range(length))
    return np.linspace(0, length - 1, num=limit, dtype=np.int64).tolist()


def proxy_metric_mask(npz_path: str, zero_rate_threshold: float = 0.999, std_eps: float = 1e-8) -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(npz_path, allow_pickle=True)
    series = data["series"].astype(np.float32)
    metric_names = data["metric_names"].astype(str).tolist() if "metric_names" in data else [f"metric_{i}" for i in range(series.shape[-1])]
    finite = np.isfinite(series)
    finite_values = np.where(finite, series, np.nan)
    std = np.nanstd(finite_values, axis=(0, 1))
    zero_rate = np.nanmean(np.abs(finite_values) <= 1e-8, axis=(0, 1))
    mask = (std > std_eps) & (zero_rate < zero_rate_threshold)
    if not bool(mask.any()):
        mask = np.ones(series.shape[-1], dtype=bool)
    included = [metric_names[idx] for idx, keep in enumerate(mask) if keep]
    excluded = [metric_names[idx] for idx, keep in enumerate(mask) if not keep]
    return mask.astype(bool), included, excluded


def build_slice_indices(
    dataset,
    candidate_indices: list[int],
    top_fraction: float,
    min_windows: int,
    max_windows: int | None,
    metric_mask: np.ndarray | None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    rows = []
    for local_idx, dataset_idx in enumerate(candidate_indices):
        x, _ = dataset[dataset_idx]
        scores = window_proxy_scores(x.numpy(), metric_mask)
        entity_idx, start_t = dataset.indices[dataset_idx]
        rows.append({"window_idx": local_idx, "dataset_idx": dataset_idx, "entity_idx": entity_idx, "start_t": start_t, **scores})
    frame = pd.DataFrame(rows)

    selections: dict[str, np.ndarray] = {}
    degraded_candidates: dict[str, np.ndarray] = {}
    for proxy in PROXY_SLICES:
        scores = frame[proxy].to_numpy(dtype=np.float64)
        order = np.argsort(-scores)
        positive = order[scores[order] > 0]
        candidate_k = max(min_windows, int(np.ceil(len(frame) * top_fraction)))
        candidate = positive[:candidate_k] if len(positive) >= min_windows else order[: min(candidate_k, len(order))]
        selected = candidate
        if max_windows is not None and max_windows > 0:
            selected = selected[:max_windows]
        degraded_candidates[proxy] = np.asarray(candidate, dtype=np.int64)
        selections[proxy] = np.asarray(selected, dtype=np.int64)

    degraded_union = set()
    for candidate in degraded_candidates.values():
        degraded_union.update(int(idx) for idx in candidate)

    normal_selections: dict[str, np.ndarray] = {}
    for proxy in PROXY_SLICES:
        scores = frame[proxy].to_numpy(dtype=np.float64)
        low_order = np.argsort(scores)
        normal = []
        for idx in low_order:
            if int(idx) not in degraded_union:
                normal.append(int(idx))
            if len(normal) >= len(selections[proxy]):
                break
        normal_selections[proxy] = np.asarray(normal, dtype=np.int64)
    return frame, selections, normal_selections, degraded_candidates


def target_stats(npz_path: str, data_cfg: DataConfig) -> tuple[float, float]:
    raw, _, _ = load_npz_cloudops(data_cfg)
    train_end = int(raw.shape[1] * data_cfg.train_ratio)
    _, scaler = normalize_by_train(raw, train_end)
    target_mean = float(scaler.mean_[data_cfg.target_metric])
    target_scale = float(scaler.scale_[data_cfg.target_metric])
    return target_mean, target_scale


def train_model(
    name: str,
    bundle,
    data_cfg: DataConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    train_dataset,
    val_dataset,
) -> nn.Module:
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
    stress_cfg = StressConfig(scenario="clean")
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
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device))
                val_losses.append(float(torch.mean((pred.cpu() - y) ** 2)))
        val_loss = float(np.mean(val_losses))
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


def predict_clean(model: nn.Module, dataset, train_cfg: TrainConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(dataset, train_cfg.batch_size, False, train_cfg.num_workers)
    preds = []
    trues = []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def score_subset(
    pred: np.ndarray,
    true: np.ndarray,
    indices: np.ndarray,
    target_mean: float,
    target_scale: float,
    capacity_cfg: CapacityConfig,
) -> dict[str, float]:
    sub_pred = pred[indices]
    sub_true = true[indices]
    quality = mse_mae(sub_pred, sub_true)
    pred_raw = sub_pred * target_scale + target_mean
    true_raw = sub_true * target_scale + target_mean
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


def format_model(name: str) -> str:
    return {"race_dlinear": "RACE-DLinear", "dlinear": "DLinear", "patchtst": "PatchTST-lite"}.get(name, name)


def write_latex_table(winners: pd.DataFrame, path: Path) -> None:
    def escape(value: object) -> str:
        text = str(value)
        replacements = {
            "\\": r"\textbackslash{}",
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        return "".join(replacements.get(ch, ch) for ch in text)

    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    rows = []
    grouped = (
        winners.groupby("dataset", as_index=False)
        .agg(
            proxy_slices=("proxy_slice", "count"),
            min_windows=("window_count", "min"),
            normal_to_slice_mse_flips=("normal_to_slice_mse_flip", "sum"),
            normal_to_slice_capacity_flips=("normal_to_slice_capacity_flip", "sum"),
            objective_flips=("mse_vs_capacity_flip", "sum"),
        )
        .sort_values("dataset")
    )
    for _, row in grouped.iterrows():
        rows.append(
            [
                source_labels.get(row["dataset"], row["dataset"]),
                int(row["proxy_slices"]),
                int(row["min_windows"]),
                int(row["normal_to_slice_mse_flips"]),
                int(row["normal_to_slice_capacity_flips"]),
                int(row["objective_flips"]),
            ]
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\tiny",
        r"\caption{Natural degradation proxy slices without synthetic stress injection. Full per-slice winners are in the artifact CSV.}",
        r"\label{tab:natural-degradation-slices}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Source & Slices & N/slice & MSE flips & Cap. flips & Obj. flips \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate natural proxy slices without synthetic stress injection.")
    parser.add_argument("--output-dir", default="outputs/natural_proxy_slice")
    parser.add_argument("--paper-table-dir", default="outputs/paper_tables")
    parser.add_argument("--latex-output", default="paper/tables/natural_degradation_slices.tex")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top-fraction", type=float, default=0.05)
    parser.add_argument("--min-windows", type=int, default=128)
    parser.add_argument("--max-slice-windows", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-train-windows", type=int, default=8192)
    parser.add_argument("--max-val-windows", type=int, default=2048)
    parser.add_argument("--max-test-windows", type=int, default=20000)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paper_table_dir = Path(args.paper_table_dir)
    paper_table_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    train_cfg = TrainConfig(
        models=MODELS,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=2,
        device=args.device,
        train_stress=False,
        calibrate=False,
    )
    capacity_cfg = CapacityConfig()

    metric_rows = []
    winner_rows = []
    slice_rows = []

    for dataset_name, npz_path in DATASETS.items():
        data_cfg = DataConfig(source="npz", npz_path=npz_path, input_len=96, pred_len=24, target_metric=0)
        bundle = build_window_bundle(data_cfg, args.seed)
        metric_mask, included_metrics, excluded_metrics = proxy_metric_mask(npz_path)
        target_mean, target_scale = target_stats(npz_path, data_cfg)
        train_indices = deterministic_subset_indices(len(bundle.train), args.max_train_windows)
        val_indices = deterministic_subset_indices(len(bundle.val), args.max_val_windows)
        test_indices = deterministic_subset_indices(len(bundle.test_clean), args.max_test_windows)
        train_dataset = Subset(bundle.train, train_indices)
        val_dataset = Subset(bundle.val, val_indices)
        test_dataset = Subset(bundle.test_clean, test_indices)
        score_frame, selections, normal_selections, degraded_candidates = build_slice_indices(
            bundle.test_clean,
            test_indices,
            args.top_fraction,
            args.min_windows,
            args.max_slice_windows,
            metric_mask,
        )
        score_frame.insert(0, "dataset", dataset_name)
        score_frame["proxy_included_metrics"] = ",".join(included_metrics)
        score_frame["proxy_excluded_metrics"] = ",".join(excluded_metrics)
        slice_rows.append(score_frame)

        predictions = {}
        trues = None
        latencies = {}
        for model_name in MODELS:
            model = train_model(model_name, bundle, data_cfg, train_cfg, device, train_dataset, val_dataset)
            pred, true = predict_clean(model, test_dataset, train_cfg, device)
            predictions[model_name] = pred
            trues = true
            sample_x, _ = next(iter(make_loader(test_dataset, train_cfg.batch_size, False, train_cfg.num_workers)))
            sample_x = sample_x[: min(32, sample_x.shape[0])]
            latencies[model_name] = measure_latency(
                model,
                sample_x,
                warmup=train_cfg.latency_warmup,
                iters=train_cfg.latency_iters,
                device=device,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

        assert trues is not None
        all_indices = np.arange(len(test_dataset), dtype=np.int64)
        global_scores = {}
        for model_name, pred in predictions.items():
            score = score_subset(pred, trues, all_indices, target_mean, target_scale, capacity_cfg)
            global_scores[model_name] = score
            metric_rows.append(
                {
                    "dataset": dataset_name,
                    "proxy_slice": "all_test_windows",
                    "model": model_name,
                    "window_count": len(all_indices),
                    **score,
                    **latencies[model_name],
                    "params": np.nan,
                }
            )

        global_best_mse = min(global_scores, key=lambda m: (global_scores[m]["mse"], m))
        global_best_capacity = min(global_scores, key=lambda m: (global_scores[m]["capacity_cost"], m))

        for proxy_name, indices in selections.items():
            slice_scores = {}
            normal_scores = {}
            normal_indices = normal_selections[proxy_name]
            for model_name, pred in predictions.items():
                normal_score = score_subset(pred, trues, normal_indices, target_mean, target_scale, capacity_cfg)
                score = score_subset(pred, trues, indices, target_mean, target_scale, capacity_cfg)
                normal_scores[model_name] = normal_score
                slice_scores[model_name] = score
                metric_rows.append(
                    {
                        "dataset": dataset_name,
                        "proxy_slice": proxy_name,
                        "slice_kind": "natural_degraded",
                        "model": model_name,
                        "window_count": len(indices),
                        **score,
                        **latencies[model_name],
                        "params": np.nan,
                    }
                )
                metric_rows.append(
                    {
                        "dataset": dataset_name,
                        "proxy_slice": proxy_name,
                        "slice_kind": "matched_normal",
                        "model": model_name,
                        "window_count": len(normal_indices),
                        **normal_score,
                        **latencies[model_name],
                        "params": np.nan,
                    }
                )
            best_mse = min(slice_scores, key=lambda m: (slice_scores[m]["mse"], m))
            best_capacity = min(slice_scores, key=lambda m: (slice_scores[m]["capacity_cost"], m))
            normal_best_mse = min(normal_scores, key=lambda m: (normal_scores[m]["mse"], m))
            normal_best_capacity = min(normal_scores, key=lambda m: (normal_scores[m]["capacity_cost"], m))
            cap_rank = sorted(slice_scores, key=lambda m: (slice_scores[m]["capacity_cost"], m))
            mse_rank = sorted(slice_scores, key=lambda m: (slice_scores[m]["mse"], m))
            winner_rows.append(
                {
                    "dataset": dataset_name,
                    "proxy_slice": proxy_name,
                    "proxy_label": PROXY_LABELS.get(proxy_name, proxy_name),
                    "window_count": len(indices),
                    "normal_window_count": len(normal_indices),
                    "top_fraction": args.top_fraction,
                    "max_slice_windows": args.max_slice_windows,
                    "top_degraded_candidate_count": len(degraded_candidates[proxy_name]),
                    "proxy_excluded_metrics": ",".join(excluded_metrics),
                    "global_best_mse_model": global_best_mse,
                    "global_best_capacity_model": global_best_capacity,
                    "normal_best_mse_model": normal_best_mse,
                    "normal_best_capacity_model": normal_best_capacity,
                    "slice_best_mse_model": best_mse,
                    "slice_best_capacity_model": best_capacity,
                    "normal_to_slice_mse_flip": best_mse != normal_best_mse,
                    "normal_to_slice_capacity_flip": best_capacity != normal_best_capacity,
                    "mse_ranking_changed": best_mse != global_best_mse,
                    "capacity_ranking_changed": best_capacity != global_best_capacity,
                    "mse_vs_capacity_flip": best_mse != best_capacity,
                    "normal_best_mse": normal_scores[normal_best_mse]["mse"],
                    "normal_best_capacity_cost": normal_scores[normal_best_capacity]["capacity_cost"],
                    "slice_best_mse": slice_scores[best_mse]["mse"],
                    "slice_best_capacity_cost": slice_scores[best_capacity]["capacity_cost"],
                    "capacity_gap": slice_scores[cap_rank[1]]["capacity_cost"] - slice_scores[cap_rank[0]]["capacity_cost"]
                    if len(cap_rank) > 1
                    else 0.0,
                    "mse_gap": slice_scores[mse_rank[1]]["mse"] - slice_scores[mse_rank[0]]["mse"] if len(mse_rank) > 1 else 0.0,
                }
            )

    metrics = pd.DataFrame(metric_rows)
    winners = pd.DataFrame(winner_rows)
    slices = pd.concat(slice_rows, ignore_index=True)
    summary = (
        winners.groupby("dataset", as_index=False)
        .agg(
            proxy_slices=("proxy_slice", "count"),
            normal_to_slice_mse_flips=("normal_to_slice_mse_flip", "sum"),
            normal_to_slice_capacity_flips=("normal_to_slice_capacity_flip", "sum"),
            mse_ranking_changes=("mse_ranking_changed", "sum"),
            capacity_ranking_changes=("capacity_ranking_changed", "sum"),
            mse_capacity_flips=("mse_vs_capacity_flip", "sum"),
        )
        .assign(
            any_ranking_change=lambda x: (
                x["normal_to_slice_mse_flips"]
                + x["normal_to_slice_capacity_flips"]
                + x["mse_ranking_changes"]
                + x["capacity_ranking_changes"]
                + x["mse_capacity_flips"]
            )
            > 0,
        )
    )

    metrics.to_csv(output_dir / "natural_proxy_slice_metrics.csv", index=False)
    winners.to_csv(output_dir / "natural_proxy_slice_winners.csv", index=False)
    summary.to_csv(output_dir / "natural_proxy_slice_summary.csv", index=False)
    slices.to_csv(output_dir / "natural_proxy_slice_window_scores.csv", index=False)
    metrics.to_csv(output_dir / "natural_slice_summary.csv", index=False)
    winners.to_csv(output_dir / "natural_slice_winners.csv", index=False)
    summary.to_csv(output_dir / "natural_slice_dataset_summary.csv", index=False)
    slices.to_csv(output_dir / "natural_slice_window_scores.csv", index=False)
    write_latex_table(winners, output_dir / "natural_degradation_slices.tex")
    winners.to_csv(paper_table_dir / "table_natural_degradation_slices.csv", index=False)
    if args.latex_output:
        latex_output = Path(args.latex_output)
        latex_output.parent.mkdir(parents=True, exist_ok=True)
        write_latex_table(winners, latex_output)

    useful = bool(
        (
            winners["normal_to_slice_mse_flip"]
            | winners["normal_to_slice_capacity_flip"]
            | winners["mse_ranking_changed"]
            | winners["capacity_ranking_changed"]
            | winners["mse_vs_capacity_flip"]
        ).any()
    )
    md = [
        "# Natural Proxy Slice Evaluation Decision",
        "",
        f"- Seed: {args.seed}",
        f"- Top fraction: {args.top_fraction}",
        f"- Minimum windows per slice: {args.min_windows}",
        f"- Maximum windows per slice: {args.max_slice_windows}",
        "- Scope: Alibaba 2018 and Salesforce/Borg, clean inputs only, no synthetic stress injection.",
        "- Proxy scoring excludes globally all-zero / near-constant metrics.",
        "- Matched normal windows: low-proxy test windows from the same source, equal count when available, excluding the union of all four top-fraction degraded candidate sets before the per-slice cap.",
        "- Interpretation: natural proxy slices are not labeled production incidents.",
        "",
        "## Reproducibility and Fairness Confirmation",
        "",
        "- Salesforce/Borg all-zero `dynamic_4` is excluded from every natural proxy score by the global all-zero / near-constant metric mask.",
        "- Matched normal windows are sampled from low-proxy windows and exclude the union of all four proxy families' top-fraction degraded candidate windows, not only the finally selected capped degraded windows.",
        "- Each degraded slice is selected from the top-fraction proxy windows and capped by `--max-slice-windows`.",
        "- Degraded and matched normal slices are drawn only from `bundle.test_clean`; slice membership is used only for evaluation after training and is not used for training, hyperparameter tuning, early stopping, or model selection.",
        "- Capacity proxy uses `CapacityConfig()` defaults: `h=0.15`, `cu=5.0`, `co=1.0`, `epsilon=0.05`; `DataConfig(... target_metric=0)` makes accuracy and capacity use the same target.",
        "- `table_natural_degradation_slices.csv` and the LaTeX table are generated by this script through `--paper-table-dir` and `--latex-output`; they are not hand-written.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Decision",
        "",
        "- Go/useful artifact evidence: yes" if useful else "- No-use for main text: no ranking change detected",
        "- Recommended paper use: artifact-first; add a short main-text sentence only if space allows.",
        "",
        "## Winner table",
        "",
        winners[
            [
                "dataset",
                "proxy_slice",
                "window_count",
                "normal_best_mse_model",
                "normal_best_capacity_model",
                "global_best_mse_model",
                "global_best_capacity_model",
                "slice_best_mse_model",
                "slice_best_capacity_model",
                "normal_to_slice_mse_flip",
                "normal_to_slice_capacity_flip",
                "mse_ranking_changed",
                "capacity_ranking_changed",
                "mse_vs_capacity_flip",
            ]
        ].to_markdown(index=False),
    ]
    (output_dir / "natural_proxy_slice_decision.md").write_text("\n".join(md), encoding="utf-8")
    (output_dir / "natural_slice_report.md").write_text("\n".join(md), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "useful": useful, "rows": len(metrics)}, indent=2))


if __name__ == "__main__":
    main()
