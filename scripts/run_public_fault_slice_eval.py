from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
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
TIME_HINTS = ["timestamp", "time", "datetime", "date", "ts"]
LABEL_HINTS = ["fault", "anomaly", "label", "phase", "status", "window_type"]
ENTITY_HINTS = ["service", "pod", "container", "instance", "node", "host", "entity"]


@dataclass
class CandidateFile:
    logical_path: str
    local_path: Path
    source_zip: str | None = None
    inject_time_path: Path | None = None


def deterministic_subset_indices(length: int, limit: int | None) -> list[int]:
    if limit is None or limit <= 0 or length <= limit:
        return list(range(length))
    return np.linspace(0, length - 1, num=limit, dtype=np.int64).tolist()


def collect_candidate_files(input_path: Path, work_dir: Path) -> list[CandidateFile]:
    candidates: list[CandidateFile] = []
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        extract_dir = work_dir / input_path.stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(input_path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith((".csv", ".parquet", "metrics.json"))]
            for name in names:
                target = extract_dir / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(name) as src, target.open("wb") as dst:
                        dst.write(src.read())
                inject_path = target.parent / "inject_time.txt"
                candidates.append(CandidateFile(name, target, input_path.name, inject_path if inject_path.exists() else None))
    elif input_path.is_dir():
        for path in input_path.rglob("*"):
            if path.suffix.lower() in {".csv", ".parquet"} or path.name.lower() == "metrics.json":
                inject_path = path.parent / "inject_time.txt"
                candidates.append(CandidateFile(str(path.relative_to(input_path)), path, inject_time_path=inject_path if inject_path.exists() else None))
    return candidates


def read_frame(path: Path, max_rows: int | None) -> pd.DataFrame | None:
    try:
        if path.name.lower() == "metrics.json":
            frame = read_metrics_json(path)
        elif path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
    except Exception:
        return None
    if max_rows and len(frame) > max_rows:
        frame = frame.iloc[:max_rows].copy()
    return frame


def read_metrics_json(path: Path) -> pd.DataFrame:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = []

    def add_series(name: str, values) -> None:
        if isinstance(values, dict):
            iterable = values.items()
        elif isinstance(values, list):
            iterable = enumerate(values)
        else:
            return
        for t, value in iterable:
            if isinstance(value, dict):
                row = {"timestamp": t, "entity": name}
                for k, v in value.items():
                    row[str(k)] = v
                rows.append(row)
            else:
                rows.append({"timestamp": t, "entity": "all", name: value})

    if isinstance(raw, dict):
        for key, values in raw.items():
            if isinstance(values, dict):
                for sub_key, sub_values in values.items():
                    add_series(f"{key}:{sub_key}", sub_values)
            else:
                add_series(str(key), values)
    elif isinstance(raw, list):
        for idx, item in enumerate(raw):
            if isinstance(item, dict):
                row = {"timestamp": item.get("timestamp", item.get("time", idx)), "entity": item.get("service", item.get("entity", "all"))}
                row.update(item)
                rows.append(row)
    if not rows:
        raise ValueError("metrics.json did not contain recognizable time series")
    frame = pd.DataFrame(rows)
    return frame


def infer_time_col(frame: pd.DataFrame) -> str | None:
    lowered = {c.lower(): c for c in frame.columns}
    for hint in TIME_HINTS:
        for key, original in lowered.items():
            if hint in key:
                return original
    return None


def infer_entity_col(frame: pd.DataFrame) -> str | None:
    lowered = {c.lower(): c for c in frame.columns}
    for hint in ENTITY_HINTS:
        for key, original in lowered.items():
            if hint in key and frame[original].nunique(dropna=True) > 1:
                return original
    return None


def infer_label_col(frame: pd.DataFrame) -> str | None:
    lowered = {c.lower(): c for c in frame.columns}
    for hint in LABEL_HINTS:
        for key, original in lowered.items():
            if hint in key and frame[original].nunique(dropna=True) <= 20:
                return original
    return None


def normalize_time(frame: pd.DataFrame, time_col: str) -> pd.Series:
    raw = frame[time_col]
    if np.issubdtype(raw.dtype, np.number):
        values = pd.to_numeric(raw, errors="coerce")
        if values.dropna().median() > 1e11:
            return pd.to_datetime(values, unit="ms", errors="coerce")
        if values.dropna().median() > 1e8:
            return pd.to_datetime(values, unit="s", errors="coerce")
        return values
    return pd.to_datetime(raw, errors="coerce")


def read_inject_time(candidate: CandidateFile) -> float | None:
    if candidate.inject_time_path is None or not candidate.inject_time_path.exists():
        return None
    try:
        text = candidate.inject_time_path.read_text(encoding="utf-8", errors="ignore").strip()
        return float(text.split()[0])
    except Exception:
        return None


def extract_fault_type(logical_path: str, label_values: pd.Series | None) -> str:
    text = logical_path.lower()
    for key in ["cpu", "memory", "network", "latency", "delay", "loss", "io", "pod", "node", "service"]:
        if key in text:
            return key
    if label_values is not None and not label_values.empty:
        values = [str(v) for v in label_values.dropna().astype(str).unique()[:3]]
        if values:
            return "|".join(values)
    return "unknown"


def frame_to_series(frame: pd.DataFrame, logical_path: str, input_len: int, pred_len: int, max_metrics: int) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict] | None:
    time_col = infer_time_col(frame)
    if time_col is None:
        return None
    frame = frame.copy()
    frame["_time"] = normalize_time(frame, time_col)
    frame = frame.dropna(subset=["_time"]).sort_values("_time")
    if frame.empty:
        return None

    entity_col = infer_entity_col(frame)
    label_col = infer_label_col(frame)
    excluded = {time_col, "_time"}
    if entity_col:
        excluded.add(entity_col)
    if label_col:
        excluded.add(label_col)
    numeric_cols = []
    for col in frame.columns:
        if col in excluded:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        finite_rate = np.isfinite(values).mean()
        if finite_rate > 0.7 and values.nunique(dropna=True) > 3:
            numeric_cols.append(col)
    if not numeric_cols:
        return None
    numeric_cols = numeric_cols[:max_metrics]

    if entity_col is None:
        entity_values = pd.Series(["all"] * len(frame), index=frame.index)
        frame["_entity"] = entity_values
        entity_col = "_entity"

    times = np.asarray(sorted(frame["_time"].dropna().unique()))
    if len(times) < input_len + pred_len + 8:
        return None
    entities = [str(v) for v in frame[entity_col].dropna().astype(str).unique()[:16]]
    arrays = []
    kept_entities = []
    for entity in entities:
        sub = frame[frame[entity_col].astype(str) == entity].set_index("_time").sort_index()
        values = sub.reindex(times)[numeric_cols].apply(pd.to_numeric, errors="coerce")
        values = values.interpolate(limit_direction="both").ffill().bfill()
        arr = values.to_numpy(dtype=np.float32)
        if np.isfinite(arr).mean() > 0.95 and np.nanstd(arr[:, 0]) > 1e-8:
            arrays.append(arr)
            kept_entities.append(entity)
    if not arrays:
        return None
    series = np.stack(arrays, axis=0).astype(np.float32)
    meta = {
        "time_col": time_col,
        "entity_col": None if entity_col == "_entity" else entity_col,
        "label_col": label_col,
        "fault_type": extract_fault_type(logical_path, frame[label_col] if label_col else None),
        "n_times": len(times),
        "n_entities": len(kept_entities),
        "n_metrics": len(numeric_cols),
    }
    return series, times, numeric_cols, kept_entities, meta


def train_model(name: str, series_norm: np.ndarray, metric_names: list[str], cfg: DataConfig, train_cfg: TrainConfig, device: torch.device) -> nn.Module:
    train_end = int(series_norm.shape[1] * cfg.train_ratio)
    val_end = int(series_norm.shape[1] * (cfg.train_ratio + cfg.val_ratio))
    train_ds = WindowDataset(series_norm, cfg.input_len, cfg.pred_len, cfg.target_metric, 0, train_end)
    val_ds = WindowDataset(series_norm, cfg.input_len, cfg.pred_len, cfg.target_metric, train_end - cfg.input_len, val_end)
    train_ds = Subset(train_ds, deterministic_subset_indices(len(train_ds), 2048))
    val_ds = Subset(val_ds, deterministic_subset_indices(len(val_ds), 512))
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
            pred = model(x)
            loss = criterion(pred, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        model.eval()
        vals = []
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
        "under_provision_rate": cap.under_rate,
        "peak_miss": cap.under_rate,
    }


def locate_window_dataset(series_norm: np.ndarray, cfg: DataConfig, times: np.ndarray, inject_time: float | None, window_type: str) -> WindowDataset | None:
    n_steps = series_norm.shape[1]
    if inject_time is None:
        train_end = int(n_steps * 0.65)
        val_end = int(n_steps * 0.80)
        if window_type == "normal":
            start = max(0, train_end - cfg.input_len)
            end = val_end
        else:
            start = max(0, val_end - cfg.input_len)
            end = n_steps
    else:
        idx = int(np.searchsorted(times, inject_time))
        if window_type == "normal":
            end = max(cfg.input_len + cfg.pred_len, idx - cfg.pred_len)
            start = max(0, end - 512)
        else:
            start = max(0, idx - cfg.input_len)
            end = min(n_steps, idx + 512)
    if end - start < cfg.input_len + cfg.pred_len:
        return None
    return WindowDataset(series_norm, cfg.input_len, cfg.pred_len, cfg.target_metric, start, end)


def evaluate_case(candidate: CandidateFile, frame: pd.DataFrame, args, device: torch.device) -> tuple[list[dict], dict]:
    parsed = frame_to_series(frame, candidate.logical_path, args.input_len, args.pred_len, args.max_metrics)
    if parsed is None:
        return [], {"status": "skipped", "reason": "cannot_parse_continuous_numeric_metrics"}
    series, times, metric_names, entity_ids, meta = parsed
    if series.shape[1] < args.input_len + args.pred_len + 8:
        return [], {"status": "skipped", "reason": "insufficient_length", **meta}
    train_end = int(series.shape[1] * 0.65)
    series_norm, scaler = normalize_by_train(series, train_end)
    cfg = DataConfig(source="npz", input_len=args.input_len, pred_len=args.pred_len, target_metric=0)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, patience=1, device=args.device, latency_iters=12, latency_warmup=4)
    cap_cfg = CapacityConfig()

    inject_time = read_inject_time(candidate)
    normal_ds = locate_window_dataset(series_norm, cfg, times, inject_time, "normal")
    fault_ds = locate_window_dataset(series_norm, cfg, times, inject_time, "fault")
    if normal_ds is None or fault_ds is None:
        return [], {"status": "skipped", "reason": "cannot_build_normal_fault_windows", **meta}
    normal_ds = Subset(normal_ds, deterministic_subset_indices(len(normal_ds), args.max_windows))
    fault_ds = Subset(fault_ds, deterministic_subset_indices(len(fault_ds), args.max_windows))

    rows = []
    for model_name in MODELS:
        model = train_model(model_name, series_norm, metric_names, cfg, train_cfg, device)
        sample = torch.zeros(1, cfg.input_len, len(metric_names), dtype=torch.float32)
        latency = measure_latency(model, sample, train_cfg.latency_warmup, train_cfg.latency_iters, device)
        for window_type, ds in [("normal", normal_ds), ("fault", fault_ds)]:
            pred, true = predict_dataset(model, ds, train_cfg, device)
            metrics = score(pred, true, float(scaler.mean_[0]), float(scaler.scale_[0]), cap_cfg)
            rows.append(
                {
                    "dataset": "RCAEval",
                    "system": infer_system(candidate.logical_path),
                    "case_id": sanitize_case_id(candidate.logical_path),
                    "fault_type": meta["fault_type"],
                    "source_file": candidate.logical_path,
                    "window_type": window_type,
                    "model": model_name,
                    "eval_windows": len(ds),
                    "n_entities": meta["n_entities"],
                    "n_steps": meta["n_times"],
                    "n_metrics": meta["n_metrics"],
                    "p95_latency_ms": latency["latency_p95_ms"],
                    **metrics,
                }
            )
    return rows, {"status": "used", **meta}


def infer_system(path: str) -> str:
    low = path.lower()
    if "online" in low or "boutique" in low or "ob" in low:
        return "online_boutique"
    if "sock" in low or "ss" in low:
        return "sock_shop"
    if "train" in low or "tt" in low:
        return "train_ticket"
    return "unknown"


def sanitize_case_id(path: str) -> str:
    parts = Path(path).parts
    if len(parts) >= 3:
        return "_".join(parts[-3:-1])
    stem = Path(path).stem
    match = re.search(r"(case[_-]?\d+|\d{3,})", path, flags=re.IGNORECASE)
    return match.group(1) if match else stem[:80]


def summarize_winners(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (case_id, system, fault_type), group in metrics.groupby(["case_id", "system", "fault_type"]):
        normal = group[group["window_type"] == "normal"]
        fault = group[group["window_type"] == "fault"]
        if normal.empty or fault.empty:
            continue
        rows.append(
            {
                "dataset": "RCAEval",
                "system": system,
                "case_id": case_id,
                "fault_type": fault_type,
                "normal_best_mse": normal.sort_values("mse").iloc[0]["model"],
                "normal_best_capacity": normal.sort_values("capacity_cost").iloc[0]["model"],
                "fault_best_mse": fault.sort_values("mse").iloc[0]["model"],
                "fault_best_capacity": fault.sort_values("capacity_cost").iloc[0]["model"],
                "ranking_changed": normal.sort_values("mse").iloc[0]["model"] != fault.sort_values("mse").iloc[0]["model"],
                "fault_mse_capacity_disagreement": fault.sort_values("mse").iloc[0]["model"] != fault.sort_values("capacity_cost").iloc[0]["model"],
            }
        )
    return pd.DataFrame(rows)


def write_decision(output_dir: Path, metrics: pd.DataFrame, winners: pd.DataFrame, skipped: list[dict]) -> None:
    lines = [
        "# Public Fault-Injection Slice Decision",
        "",
        "- Scope: public fault-injection slice evidence, not production incident validation.",
        "- Source priority: RCAEval RE1 metrics-style files when available.",
        "",
    ]
    if metrics.empty:
        lines += [
            "## Decision",
            "",
            "No-use / blocked.",
            "",
            "No parseable continuous numeric metric cases were found. Keep current controlled-stress paper unchanged.",
        ]
    else:
        n_cases = winners["case_id"].nunique() if not winners.empty else 0
        changed = int(winners["ranking_changed"].sum()) if not winners.empty else 0
        disagree = int(winners["fault_mse_capacity_disagreement"].sum()) if not winners.empty else 0
        if n_cases >= 20 and (changed > 0 or disagree > 0):
            decision = "Go for a short main-text sentence or tiny table."
        elif n_cases >= 10 and (changed > 0 or disagree > 0):
            decision = "Artifact-only unless space is available."
        else:
            decision = "Artifact-only / no-use."
        lines += [
            "## Decision",
            "",
            decision,
            "",
            "## Summary",
            "",
            f"- Parsed cases: {n_cases}",
            f"- Normal-to-fault MSE ranking changes: {changed}",
            f"- Fault-window MSE-vs-capacity disagreements: {disagree}",
            f"- Metric rows: {len(metrics)}",
        ]
    lines += ["", "## Skipped Files", "", f"- Skipped candidates: {len(skipped)}"]
    for row in skipped[:20]:
        lines.append(f"- `{row.get('file', '')}`: {row.get('reason', row.get('status', 'unknown'))}")
    (output_dir / "fault_slice_decision.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/rcaeval/RE1-OB.zip")
    parser.add_argument("--output-dir", default="outputs/public_fault_slice")
    parser.add_argument("--work-dir", default="outputs/public_fault_slice/_extract")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--max-files", type=int, default=60)
    parser.add_argument("--max-rows-per-file", type=int, default=200000)
    parser.add_argument("--max-metrics", type=int, default=8)
    parser.add_argument("--max-windows", type=int, default=512)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        (output_dir / "fault_slice_metrics.csv").write_text("", encoding="utf-8")
        (output_dir / "fault_slice_winners.csv").write_text("", encoding="utf-8")
        pd.DataFrame([{"status": "blocked", "reason": "input_not_found", "input": str(input_path)}]).to_csv(output_dir / "fault_slice_summary.csv", index=False)
        write_decision(output_dir, pd.DataFrame(), pd.DataFrame(), [{"file": str(input_path), "reason": "input_not_found"}])
        print(json.dumps({"status": "blocked", "reason": "input_not_found", "input": str(input_path)}, indent=2))
        return

    candidates = collect_candidate_files(input_path, work_dir)[: args.max_files]
    device = select_device(args.device)
    all_rows: list[dict] = []
    skipped: list[dict] = []
    for candidate in candidates:
        frame = read_frame(candidate.local_path, args.max_rows_per_file)
        if frame is None:
            skipped.append({"file": candidate.logical_path, "reason": "read_failed"})
            continue
        try:
            rows, status = evaluate_case(candidate, frame, args, device)
        except Exception as exc:
            rows, status = [], {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}"}
        if rows:
            all_rows.extend(rows)
        else:
            skipped.append({"file": candidate.logical_path, **status})
        if len({r["case_id"] for r in all_rows}) >= 50:
            break

    metrics = pd.DataFrame(all_rows)
    winners = summarize_winners(metrics) if not metrics.empty else pd.DataFrame()
    summary = pd.DataFrame(
        [
            {
                "candidate_files": len(candidates),
                "parsed_cases": int(winners["case_id"].nunique()) if not winners.empty else 0,
                "metric_rows": len(metrics),
                "ranking_changed_cases": int(winners["ranking_changed"].sum()) if not winners.empty else 0,
                "fault_mse_capacity_disagreement_cases": int(winners["fault_mse_capacity_disagreement"].sum()) if not winners.empty else 0,
                "skipped_files": len(skipped),
            }
        ]
    )
    metrics.to_csv(output_dir / "fault_slice_metrics.csv", index=False)
    winners.to_csv(output_dir / "fault_slice_winners.csv", index=False)
    summary.to_csv(output_dir / "fault_slice_summary.csv", index=False)
    (output_dir / "fault_slice_skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    write_decision(output_dir, metrics, winners, skipped)
    print(json.dumps(summary.iloc[0].to_dict(), indent=2))


if __name__ == "__main__":
    main()
