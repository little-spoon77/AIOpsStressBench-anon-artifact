from __future__ import annotations

import argparse
import csv
import json
import random
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from race_forecast.config import StressConfig, load_config
from race_forecast.data import build_window_bundle
from race_forecast.metrics import capacity_proxy, mse_mae
from race_forecast.models import build_model
from race_forecast.train import _calibrate_if_needed, _predict, make_loader, select_device, train_one_model


CASES = {
    "netman_missing_variables_30": {
        "config": "configs/netman_kpi_example.yaml",
        "stress": {"scenario": "missing_variables", "missing_rate": 0.3},
        "title": "NetMan KPI: telemetry outage",
        "risk": "Metric-level telemetry outage can hide correlated operational signals and sharply changes the robustness/latency tradeoff.",
        "deployment": "A clean benchmark would not reveal whether the selected model tolerates missing monitoring channels during incident response.",
    },
    "gaia_changepoint_level_shift": {
        "config": "configs/gaia_changepoint.yaml",
        "stress": {"scenario": "level_shift", "level_shift": 0.4},
        "title": "GAIA changepoint: level shift",
        "risk": "A release, migration, or workload regime change can shift the recent context before the forecasting horizon.",
        "deployment": "Offline clean MSE does not tell operators which model is stable under deployment-induced concept shift.",
    },
    "salesforce_borg_delayed_12": {
        "config": "configs/salesforce_borg_256x2048.yaml",
        "stress": {"scenario": "delayed_tail", "delay_steps": 12},
        "title": "Salesforce/Borg: delayed telemetry",
        "risk": "Recent workload telemetry can arrive late, forcing the forecaster to predict from a stale context window.",
        "deployment": "A clean benchmark cannot tell whether a model remains useful when the newest monitoring samples are unavailable.",
    },
    "netman_missing_30": {
        "config": "configs/netman_kpi_example.yaml",
        "stress": {"scenario": "missing_points", "missing_rate": 0.3},
        "title": "NetMan KPI: 30% missing points",
        "risk": "Point-wise missing telemetry creates an accuracy/latency tradeoff: stronger sequence models can be more accurate but costlier.",
        "deployment": "The lowest-error model may be inappropriate for a low-latency monitoring path if its P95 latency is much higher.",
    },
    "alibaba2018_missing_variables_30": {
        "config": "configs/alibaba2018_machine_usage.yaml",
        "stress": {"scenario": "missing_variables", "missing_rate": 0.3},
        "title": "Alibaba 2018: missing resource metrics",
        "risk": "Machine-level CPU, memory, network, and disk metrics may become partially unavailable during telemetry outages.",
        "deployment": "Resource forecasting should be evaluated under missing-variable stress, not only under clean machine-utilization traces.",
    },
}


MODEL_LABELS = {
    "dlinear": "DLinear",
    "race_dlinear": "RACE-DLinear",
    "patchtst": "PatchTST-lite",
}


COLORS = {
    "actual": "#222222",
    "dlinear": "#3B6FB6",
    "race_dlinear": "#D9822B",
    "patchtst": "#2F9E6D",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_stress(stress_values: dict[str, float | str]) -> StressConfig:
    cfg = StressConfig()
    for key, value in stress_values.items():
        setattr(cfg, key, value)
    return cfg


def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def write_markdown_table(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_overlay(case_id: str, case: dict[str, object], true: np.ndarray, preds: dict[str, np.ndarray], idx: int, output_dir: Path) -> None:
    apply_publication_style()
    x = np.arange(true.shape[1])
    fig, ax = plt.subplots(figsize=(7.2, 3.1))
    ax.plot(x, true[idx], label="Actual", color=COLORS["actual"], linewidth=2.4)
    for model_name, pred in preds.items():
        ax.plot(
            x,
            pred[idx],
            label=MODEL_LABELS.get(model_name, model_name),
            color=COLORS.get(model_name),
            linewidth=1.7,
            alpha=0.92,
        )
    ax.set_title(str(case["title"]))
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("Target value")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.24), frameon=False)
    fig.tight_layout(pad=0.8)
    fig.savefig(output_dir / f"{case_id}_overlay.png", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_dir / f"{case_id}_overlay.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def run_case(case_id: str, case: dict[str, object], args: argparse.Namespace) -> None:
    cfg = load_config(case["config"])
    cfg = deepcopy(cfg)
    cfg.stress = clean_stress(case["stress"])
    cfg.train.models = args.models
    cfg.train.epochs = args.epochs
    cfg.train.latency_iters = args.latency_iters
    cfg.train.latency_warmup = args.latency_warmup
    cfg.train.device = args.device

    set_seed(args.seed)
    device = select_device(cfg.train.device)
    bundle = build_window_bundle(cfg.data, args.seed)
    n_metrics = len(bundle.metric_names)
    val_loader = make_loader(bundle.val, cfg.train.batch_size, False, cfg.train.num_workers)
    test_loader = make_loader(bundle.test_clean, cfg.train.batch_size, False, cfg.train.num_workers)

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw" / case_id
    raw_dir.mkdir(parents=True, exist_ok=True)

    preds: dict[str, np.ndarray] = {}
    true_raw_ref: np.ndarray | None = None
    rows = []
    for model_name in args.models:
        set_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model = build_model(
            model_name,
            input_len=cfg.data.input_len,
            pred_len=cfg.data.pred_len,
            n_metrics=n_metrics,
            target_metric=cfg.data.target_metric,
        )
        metrics = train_one_model(
            model=model,
            name=model_name,
            train_dataset=bundle.train,
            val_dataset=bundle.val,
            test_dataset=bundle.test_clean,
            train_cfg=cfg.train,
            stress_cfg=cfg.stress,
            capacity_cfg=cfg.capacity,
            output_dir=raw_dir / model_name,
            device=device,
            target_mean=float(bundle.scaler.mean_[cfg.data.target_metric]),
            target_scale=float(bundle.scaler.scale_[cfg.data.target_metric]),
        )
        set_seed(args.seed + 17)
        pred, true = _predict(model, test_loader, cfg.stress, device)
        pred, _ = _calibrate_if_needed(model, model_name, val_loader, pred, cfg.stress, device, cfg.train.calibrate)
        pred_raw = pred * float(bundle.scaler.scale_[cfg.data.target_metric]) + float(bundle.scaler.mean_[cfg.data.target_metric])
        true_raw = true * float(bundle.scaler.scale_[cfg.data.target_metric]) + float(bundle.scaler.mean_[cfg.data.target_metric])
        preds[model_name] = pred_raw
        true_raw_ref = true_raw
        row = {
            "case_id": case_id,
            "model": MODEL_LABELS.get(model_name, model_name),
            "global_mse": metrics["mse"],
            "global_mae": metrics["mae"],
            "latency_p95_ms": metrics["latency_p95_ms"],
            "params": metrics["params"],
            "max_memory_mb": metrics["max_memory_mb"],
            "global_capacity_cost": metrics["capacity_cost"],
        }
        rows.append(row)

    if true_raw_ref is None:
        raise RuntimeError(f"No predictions generated for {case_id}")
    score = np.mean([np.mean(np.abs(pred - true_raw_ref), axis=1) for pred in preds.values()], axis=0)
    selected = int(np.argmax(score))

    local_rows = []
    for row in rows:
        model_name = next(key for key, label in MODEL_LABELS.items() if label == row["model"])
        pred = preds[model_name][selected : selected + 1]
        true = true_raw_ref[selected : selected + 1]
        quality = mse_mae(pred, true)
        capacity = capacity_proxy(
            pred,
            true,
            headroom=cfg.capacity.headroom,
            under_cost=cfg.capacity.under_cost,
            over_cost=cfg.capacity.over_cost,
            demand_floor=cfg.capacity.demand_floor,
        )
        local_rows.append(
            {
                **row,
                "selected_window": selected,
                "local_mse": quality["mse"],
                "local_mae": quality["mae"],
                "local_capacity_cost": capacity.cost,
                "local_under_rate": capacity.under_rate,
                "local_over_rate": capacity.over_rate,
            }
        )

    plot_overlay(case_id, case, true_raw_ref, preds, selected, output_dir)
    table = pd.DataFrame(local_rows).sort_values(["local_mse", "latency_p95_ms"])
    table.to_csv(output_dir / f"{case_id}_metrics.csv", index=False)
    write_markdown_table(table, output_dir / f"{case_id}_metrics.md")
    np.savez_compressed(
        output_dir / f"{case_id}_predictions.npz",
        true=true_raw_ref[selected],
        **{name: pred[selected] for name, pred in preds.items()},
    )
    note = {
        "case_id": case_id,
        "title": case["title"],
        "selected_window": selected,
        "risk": case["risk"],
        "deployment_implication": case["deployment"],
        "official_itransformer_note": "Official iTransformer is reported in summary tables. It is not overlaid here because the LTSF bridge flattens entity-metric channels and is not a per-entity target protocol.",
    }
    (output_dir / f"{case_id}_notes.json").write_text(json.dumps(note, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready multi-model case-study overlays.")
    parser.add_argument("--output-dir", default="outputs/paper_figures")
    parser.add_argument("--cases", nargs="*", default=list(CASES.keys()), choices=list(CASES.keys()))
    parser.add_argument("--models", nargs="*", default=["dlinear", "race_dlinear", "patchtst"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--latency-iters", type=int, default=24)
    parser.add_argument("--latency-warmup", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "case_index.csv"
    index_rows: dict[str, dict[str, str]] = {}
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("case_id"):
                    index_rows[row["case_id"]] = row
    for case_id in args.cases:
        case = CASES[case_id]
        index_rows[case_id] = {
            "case_id": case_id,
            "title": str(case["title"]),
            "risk": str(case["risk"]),
            "deployment": str(case["deployment"]),
        }
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "title", "risk", "deployment"])
        writer.writeheader()
        for case_id in CASES:
            if case_id in index_rows:
                writer.writerow(index_rows[case_id])

    for case_id in args.cases:
        print(f"Generating case study: {case_id}")
        run_case(case_id, CASES[case_id], args)
    print(f"Saved case-study figures to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
