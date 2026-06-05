from __future__ import annotations

import argparse
import json
import pickle
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Subset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from race_forecast.config import CapacityConfig, DataConfig, TrainConfig
from race_forecast.data import WindowDataset, normalize_by_train
from race_forecast.metrics import capacity_proxy, measure_latency, mse_mae
from race_forecast.models import build_model, is_trainable
from race_forecast.train import make_loader, select_device


MODELS = ["dlinear", "race_dlinear", "patchtst"]
FEATURE_COLUMNS = [
    "istio_latency_50",
    "istio_latency_90",
    "istio_latency_95",
    "istio_latency_99",
    "container_memory_usage_bytes",
    "container_memory_rss",
    "container_memory_working_set_bytes",
    "container_cpu_user_seconds_total",
    "container_cpu_system_seconds_total",
    "istio_request_total",
    "container_network_receive_bytes_total",
    "container_network_transmit_bytes_total",
]


class _Stub:
    def __init__(self, *args, **kwargs) -> None:
        self.__dict__.update(kwargs)

    def __setstate__(self, state) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _StubStorage(_Stub):
    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value) -> None:
        self.__dict__[key] = value


def install_torch_geometric_stubs() -> None:
    for name in ["torch_geometric", "torch_geometric.data", "torch_geometric.data.data", "torch_geometric.data.storage"]:
        mod = types.ModuleType(name)
        mod.__path__ = []
        mod.Data = _Stub
        mod.DataEdgeAttr = _StubStorage
        mod.DataTensorAttr = _StubStorage
        mod.GlobalStorage = _StubStorage
        mod.NodeStorage = _StubStorage
        mod.EdgeStorage = _StubStorage
        mod.BaseStorage = _StubStorage
        sys.modules[name] = mod


@dataclass
class FaultCase:
    experiment_id: str
    rep: int
    root_cause: str
    fault_type: str
    series: np.ndarray
    metric_names: list[str]
    entity_ids: list[str]
    normal_end: int


def deterministic_subset_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit <= 0 or length <= limit:
        return list(range(length))
    return np.linspace(0, length - 1, num=limit, dtype=np.int64).tolist()


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_re2ob(path: Path) -> list[dict]:
    install_torch_geometric_stubs()
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict) and "experiments" in obj:
        return list(obj["experiments"])
    if isinstance(obj, list):
        return obj
    raise ValueError("Unsupported RE2-OB pickle structure")


def fault_type_from_experiment(experiment_id: str) -> str:
    parts = experiment_id.split("_", 1)
    return parts[1] if len(parts) > 1 else "unknown"


def experiment_to_case(exp: dict, max_metrics: int) -> FaultCase | None:
    ts_features = exp.get("ts_features")
    if not isinstance(ts_features, dict) or not ts_features:
        return None
    entity_ids = [str(k) for k in ts_features.keys()]
    metric_names = [c for c in FEATURE_COLUMNS if c in next(iter(ts_features.values())).columns][:max_metrics]
    if not metric_names:
        return None
    arrays = []
    for entity in entity_ids:
        frame = ts_features[entity].copy()
        frame = frame.sort_values("time")
        values = frame[metric_names].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both").ffill().bfill()
        arr = values.to_numpy(dtype=np.float32)
        if not np.isfinite(arr).all():
            return None
        arrays.append(arr)
    lengths = {a.shape[0] for a in arrays}
    if len(lengths) != 1:
        return None
    series = np.stack(arrays, axis=0).astype(np.float32)
    normal_pyg_len = len(exp.get("normal_pyg", []))
    normal_end = normal_pyg_len if normal_pyg_len > 0 else series.shape[1] // 2
    return FaultCase(
        experiment_id=str(exp.get("experiment_id", "unknown")),
        rep=int(exp.get("rep", 0)),
        root_cause=str(exp.get("root_cause", "unknown")),
        fault_type=fault_type_from_experiment(str(exp.get("experiment_id", ""))),
        series=series,
        metric_names=metric_names,
        entity_ids=entity_ids,
        normal_end=normal_end,
    )


def train_model(
    name: str,
    series_norm: np.ndarray,
    metric_names: list[str],
    cfg: DataConfig,
    train_cfg: TrainConfig,
    device: torch.device,
    train_end: int,
) -> nn.Module:
    train_ds = WindowDataset(series_norm, cfg.input_len, cfg.pred_len, cfg.target_metric, 0, train_end)
    val_ds = WindowDataset(series_norm, cfg.input_len, cfg.pred_len, cfg.target_metric, 0, train_end)
    train_ds = Subset(train_ds, deterministic_subset_indices(len(train_ds), 1024))
    val_ds = Subset(val_ds, deterministic_subset_indices(len(val_ds), 256))
    model = build_model(name, cfg.input_len, cfg.pred_len, len(metric_names), cfg.target_metric).to(device)
    if not is_trainable(model):
        return model
    loader = make_loader(train_ds, train_cfg.batch_size, True, train_cfg.num_workers)
    val_loader = make_loader(val_ds, train_cfg.batch_size, False, train_cfg.num_workers)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    criterion = nn.MSELoss()
    best_state = None
    best_val = float("inf")
    for _ in range(train_cfg.epochs):
        model.train()
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        vals = []
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                vals.append(float(torch.mean((model(x.to(device)).cpu() - y) ** 2)))
        mean_val = float(np.mean(vals)) if vals else float("inf")
        if mean_val < best_val:
            best_val = mean_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def make_slice_dataset(
    series_norm: np.ndarray,
    cfg: DataConfig,
    normal_end: int,
    window_type: str,
    normal_eval_start: int | None = None,
) -> WindowDataset | None:
    n_steps = series_norm.shape[1]
    if window_type == "normal":
        if normal_eval_start is None:
            raise ValueError("normal_eval_start is required for held-out normal evaluation")
        start = normal_eval_start
        end = normal_end
    else:
        start = normal_end
        end = n_steps
    try:
        return WindowDataset(series_norm, cfg.input_len, cfg.pred_len, cfg.target_metric, start, end)
    except ValueError:
        return None


def predict_dataset(model: nn.Module, dataset: WindowDataset, train_cfg: TrainConfig, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    loader = make_loader(dataset, train_cfg.batch_size, False, train_cfg.num_workers)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device))
            preds.append(pred.detach().cpu().numpy())
            trues.append(y.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def score(pred: np.ndarray, true: np.ndarray, mean: float, scale: float, cap_cfg: CapacityConfig) -> dict[str, float]:
    quality = mse_mae(pred, true)
    pred_raw = pred * scale + mean
    true_raw = true * scale + mean
    cap = capacity_proxy(pred_raw, true_raw, cap_cfg.headroom, cap_cfg.under_cost, cap_cfg.over_cost, cap_cfg.demand_floor)
    return {
        "mse": quality["mse"],
        "mae": quality["mae"],
        "capacity_cost": cap.cost,
        "decision_cost": cap.cost,
        "under_rate": cap.under_rate,
        "peak_miss": cap.under_rate,
    }


def evaluate_case(case: FaultCase, args, device: torch.device) -> tuple[list[dict], dict]:
    min_len = args.input_len + args.pred_len
    train_end = int(case.normal_end * args.normal_train_ratio)
    normal_eval_start = train_end + args.pred_len
    if train_end < min_len or (case.normal_end - normal_eval_start) < min_len or (case.series.shape[1] - case.normal_end) < min_len:
        return [], {
            "status": "skipped",
            "reason": "insufficient_train_normal_holdout_or_fault_length",
            "experiment_id": case.experiment_id,
            "rep": case.rep,
            "normal_end": case.normal_end,
            "n_steps": case.series.shape[1],
            "train_end": train_end,
            "normal_eval_start": normal_eval_start,
        }
    series_norm, scaler = normalize_by_train(case.series, train_end)
    cfg = DataConfig(source="npz", input_len=args.input_len, pred_len=args.pred_len, target_metric=args.target_metric)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, patience=1, device=args.device, latency_iters=12, latency_warmup=4)
    cap_cfg = CapacityConfig()
    train_count_ds = WindowDataset(series_norm, cfg.input_len, cfg.pred_len, cfg.target_metric, 0, train_end)
    normal_ds = make_slice_dataset(series_norm, cfg, case.normal_end, "normal", normal_eval_start)
    fault_ds = make_slice_dataset(series_norm, cfg, case.normal_end, "fault")
    if normal_ds is None or fault_ds is None:
        return [], {"status": "skipped", "reason": "cannot_build_slice_windows", "experiment_id": case.experiment_id, "rep": case.rep}
    normal_ds = Subset(normal_ds, deterministic_subset_indices(len(normal_ds), args.max_windows))
    fault_ds = Subset(fault_ds, deterministic_subset_indices(len(fault_ds), args.max_windows))
    train_windows = len(train_count_ds)
    heldout_normal_windows = len(normal_ds)
    fault_windows = len(fault_ds)
    rows = []
    for model_name in MODELS:
        model = train_model(model_name, series_norm, case.metric_names, cfg, train_cfg, device, train_end)
        sample = torch.zeros(1, cfg.input_len, len(case.metric_names), dtype=torch.float32)
        latency = measure_latency(model, sample, train_cfg.latency_warmup, train_cfg.latency_iters, device)
        for window_type, ds in [("normal", normal_ds), ("fault", fault_ds)]:
            pred, true = predict_dataset(model, ds, train_cfg, device)
            metrics = score(pred, true, float(scaler.mean_[args.target_metric]), float(scaler.scale_[args.target_metric]), cap_cfg)
            rows.append(
                {
                    "dataset": "RE2-OB",
                    "system": "online_boutique",
                    "case_id": f"{case.experiment_id}_rep{case.rep}",
                    "fault_type": case.fault_type,
                    "root_cause": case.root_cause,
                    "window_type": window_type,
                    "model": model_name,
                    "eval_windows": len(ds),
                    "n_entities": case.series.shape[0],
                    "n_steps": case.series.shape[1],
                    "n_metrics": case.series.shape[2],
                    "target_metric_name": case.metric_names[args.target_metric],
                    "target_metric_role": "latency" if "latency" in case.metric_names[args.target_metric] else "resource_or_counter",
                    "cost_label": "decision_cost",
                    "train_window_scope": "pre_injection_only",
                    "normal_eval_scope": "held_out_pre_injection_normal",
                    "train_end": train_end,
                    "normal_eval_start": normal_eval_start,
                    "normal_eval_gap_steps": args.pred_len,
                    "scaler_fit_scope": "train_normal_only",
                    "eval_window_crosses_injection": False,
                    "train_windows": train_windows,
                    "heldout_normal_windows": heldout_normal_windows,
                    "fault_windows": fault_windows,
                    "input_len": args.input_len,
                    "pred_len": args.pred_len,
                    "p95_latency_ms": latency["latency_p95_ms"],
                    **metrics,
                }
            )
    return rows, {"status": "used", "experiment_id": case.experiment_id, "rep": case.rep}


def summarize_winners(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_id, fault_type, root_cause), group in metrics.groupby(["case_id", "fault_type", "root_cause"]):
        normal = group[group["window_type"] == "normal"]
        fault = group[group["window_type"] == "fault"]
        if normal.empty or fault.empty:
            continue
        normal_mse = normal.sort_values("mse").iloc[0]
        normal_cap = normal.sort_values("decision_cost").iloc[0]
        fault_mse = fault.sort_values("mse").iloc[0]
        fault_cap = fault.sort_values("decision_cost").iloc[0]
        fault_normal_mse_ratio = float(fault_mse["mse"]) / max(float(normal_mse["mse"]), 1e-12)
        fault_normal_cost_ratio = float(fault_cap["decision_cost"]) / max(float(normal_cap["decision_cost"]), 1e-12)
        rows.append(
            {
                "dataset": "RE2-OB",
                "system": "online_boutique",
                "case_id": case_id,
                "fault_type": fault_type,
                "root_cause": root_cause,
                "normal_best_mse": normal_mse["model"],
                "normal_best_decision_cost": normal_cap["model"],
                "fault_best_mse": fault_mse["model"],
                "fault_best_decision_cost": fault_cap["model"],
                "ranking_changed": normal_mse["model"] != fault_mse["model"],
                "fault_mse_decision_cost_disagreement": fault_mse["model"] != fault_cap["model"],
                "normal_best_mse_value": normal_mse["mse"],
                "fault_best_mse_value": fault_mse["mse"],
                "normal_best_decision_cost_value": normal_cap["decision_cost"],
                "fault_best_decision_cost_value": fault_cap["decision_cost"],
                "fault_normal_mse_ratio": fault_normal_mse_ratio,
                "fault_normal_decision_cost_ratio": fault_normal_cost_ratio,
            }
        )
    return pd.DataFrame(rows)


def write_outputs(output_dir: Path, metrics: pd.DataFrame, winners: pd.DataFrame, skipped: list[dict], args) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "fault_slice_v2_metrics.csv", index=False)
    winners.to_csv(output_dir / "fault_slice_v2_winners.csv", index=False)
    summary = {
        "available_cases": int(args.available_cases),
        "candidate_cases": args.max_cases,
        "parsed_cases": int(winners["case_id"].nunique()) if not winners.empty else 0,
        "metric_rows": int(len(metrics)),
        "ranking_changed_cases": int(winners["ranking_changed"].sum()) if not winners.empty else 0,
        "fault_mse_decision_cost_disagreement_cases": int(winners["fault_mse_decision_cost_disagreement"].sum()) if not winners.empty else 0,
        "skipped_cases": len(skipped),
        "input_len": args.input_len,
        "pred_len": args.pred_len,
        "normal_train_ratio": args.normal_train_ratio,
        "target_metric": args.target_metric,
        "target_metric_name": str(metrics["target_metric_name"].iloc[0]) if not metrics.empty else "",
        "target_metric_role": str(metrics["target_metric_role"].iloc[0]) if not metrics.empty else "",
        "training_granularity": "per_case",
        "median_train_windows_per_case": float(metrics.drop_duplicates("case_id")["train_windows"].median()) if not metrics.empty else 0.0,
        "median_heldout_normal_windows_per_case": float(metrics.drop_duplicates("case_id")["heldout_normal_windows"].median()) if not metrics.empty else 0.0,
        "median_fault_windows_per_case": float(metrics.drop_duplicates("case_id")["fault_windows"].median()) if not metrics.empty else 0.0,
        "case_selection": "first max_cases records from the RE2-OB pickle after parseability checks; remaining available records are not evaluated for this bounded probe",
    }
    pd.DataFrame([summary]).to_csv(output_dir / "fault_slice_v2_summary.csv", index=False)
    (output_dir / "fault_slice_v2_skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    n_cases = summary["parsed_cases"]
    changed = summary["ranking_changed_cases"]
    disagree = summary["fault_mse_decision_cost_disagreement_cases"]
    if n_cases >= 20 and (changed > 0 or disagree > 0):
        decision = "Go: short main-text sentence or tiny table is supportable."
    elif n_cases >= 10 and (changed > 0 or disagree > 0):
        decision = "Artifact-only: trend exists but evidence is too small for main text."
    else:
        decision = "No-use / caution: keep as artifact probe only."
    lines = [
        "# Public Fault-Injection Slice v2 Decision",
        "",
        "- Scope: public fault-injection slice sanity evidence, not production incident validation.",
        "- Source: RE2-OB PyG public Online Boutique fault-injection telemetry.",
        f"- Window protocol: input_len={args.input_len}, pred_len={args.pred_len}; short-window sanity probe because RE2-OB normal/anomaly halves have 72 snapshots.",
        f"- Target metric: index {args.target_metric} ({summary['target_metric_name']}), treated as {summary['target_metric_role']}; the asymmetric proxy is therefore reported as decision cost rather than resource capacity cost for this probe.",
        f"- Training granularity: {summary['training_granularity']}. The forecasters are trained separately for each case rather than pooling train-normal windows across cases.",
        f"- Normal holdout: the pre-injection segment is split into train-normal and held-out-normal with normal_train_ratio={args.normal_train_ratio}; held-out-normal starts one prediction horizon after the train-normal boundary.",
        "- Training/scaling scope: models and scalers use only train-normal windows. Held-out-normal and fault windows are never used for fitting.",
        "- Evaluation scope: normal evaluation windows are held-out pre-injection windows; fault evaluation windows start after the injection boundary and do not cross it.",
        f"- Window counts: median train windows per case={summary['median_train_windows_per_case']:.0f}, median held-out normal windows per case={summary['median_heldout_normal_windows_per_case']:.0f}, median fault windows per case={summary['median_fault_windows_per_case']:.0f}.",
        f"- Case scope: {summary['parsed_cases']} of {summary['available_cases']} available records are evaluated because this bounded probe uses max_cases={args.max_cases}; skipped records are reported separately.",
        "- Ratio aggregation: fault/normal MSE ratio and decision-cost ratio are computed per case from the best fault-window winner divided by the matched held-out-normal winner; compact tables report medians across cases, not ratios of global means.",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Summary",
        "",
        f"- Parsed cases: {n_cases}",
        f"- Metric rows: {summary['metric_rows']}",
        f"- Normal-to-fault MSE ranking changes: {changed}",
        f"- Fault-window MSE-vs-decision-cost disagreements: {disagree}",
        f"- Skipped cases: {len(skipped)}",
        "",
        "## Paper Use",
        "",
        "Use only as public fault-injection slice sanity evidence. Do not call it production incident validation or a replacement for the controlled stress benchmark.",
    ]
    (output_dir / "fault_slice_v2_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/re2ob_pyg/re2ob_pyg.pkl")
    parser.add_argument("--output-dir", default="outputs/public_fault_slice_v2")
    parser.add_argument("--input-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=12)
    parser.add_argument("--target-metric", type=int, default=3)
    parser.add_argument("--max-metrics", type=int, default=12)
    parser.add_argument("--max-cases", type=int, default=60)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--normal-train-ratio", type=float, default=0.6)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    set_reproducible_seed(args.seed)
    device = select_device(args.device)
    experiments = load_re2ob(Path(args.input))
    args.available_cases = len(experiments)
    rows: list[dict] = []
    skipped: list[dict] = []
    for exp in experiments[: args.max_cases]:
        case = experiment_to_case(exp, args.max_metrics)
        if case is None:
            skipped.append({"status": "skipped", "reason": "cannot_parse_case", "experiment_id": exp.get("experiment_id"), "rep": exp.get("rep")})
            continue
        case_rows, status = evaluate_case(case, args, device)
        if case_rows:
            rows.extend(case_rows)
        else:
            skipped.append(status)
    metrics = pd.DataFrame(rows)
    winners = summarize_winners(metrics) if not metrics.empty else pd.DataFrame()
    write_outputs(Path(args.output_dir), metrics, winners, skipped, args)
    print(f"wrote {len(metrics)} metric rows for {winners['case_id'].nunique() if not winners.empty else 0} cases to {args.output_dir}")


if __name__ == "__main__":
    main()
