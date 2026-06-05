from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from race_forecast.config import DataConfig


@dataclass
class WindowBundle:
    train: "WindowDataset"
    val: "WindowDataset"
    test_clean: "WindowDataset"
    scaler: StandardScaler
    metric_names: list[str]
    entity_ids: list[str]


class WindowDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        input_len: int,
        pred_len: int,
        target_metric: int,
        start: int,
        end: int,
    ) -> None:
        if series.ndim != 3:
            raise ValueError("series must have shape [entities, time, metrics]")
        self.series = series.astype(np.float32)
        self.input_len = input_len
        self.pred_len = pred_len
        self.target_metric = target_metric
        self.indices: list[tuple[int, int]] = []
        last_start = end - input_len - pred_len + 1
        for entity_idx in range(series.shape[0]):
            for t in range(max(0, start), max(0, last_start)):
                self.indices.append((entity_idx, t))
        if not self.indices:
            raise ValueError("No windows were created. Increase n_steps or reduce input_len/pred_len.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        entity_idx, t = self.indices[idx]
        x = self.series[entity_idx, t : t + self.input_len, :]
        y = self.series[
            entity_idx,
            t + self.input_len : t + self.input_len + self.pred_len,
            self.target_metric,
        ]
        return torch.from_numpy(x), torch.from_numpy(y)


def make_synthetic_cloudops(cfg: DataConfig, seed: int) -> tuple[np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    t = np.arange(cfg.n_steps, dtype=np.float32)
    daily = np.sin(2 * np.pi * t / max(1, int(24 * 60 / cfg.freq_minutes)))
    half_daily = np.sin(2 * np.pi * t / max(1, int(12 * 60 / cfg.freq_minutes)))
    weekly = np.sin(2 * np.pi * t / max(1, int(7 * 24 * 60 / cfg.freq_minutes)))
    series = np.zeros((cfg.n_entities, cfg.n_steps, cfg.n_metrics), dtype=np.float32)

    for entity_idx in range(cfg.n_entities):
        phase = rng.uniform(0, 2 * np.pi)
        scale = rng.uniform(0.5, 1.5)
        trend = rng.uniform(-0.0003, 0.0005) * t
        deploy_shift = np.zeros_like(t)
        if cfg.n_steps > 300:
            shift_start = rng.integers(cfg.n_steps // 3, cfg.n_steps - cfg.n_steps // 5)
            deploy_shift[shift_start:] = rng.uniform(-0.2, 0.35)

        workload = (
            0.45
            + 0.22 * np.sin(2 * np.pi * t / max(1, int(24 * 60 / cfg.freq_minutes)) + phase)
            + 0.09 * half_daily
            + 0.06 * weekly
            + trend
            + deploy_shift
        )
        burst_count = max(2, cfg.n_steps // 180)
        for _ in range(burst_count):
            center = rng.integers(48, max(49, cfg.n_steps - 48))
            width = rng.uniform(4, 18)
            amplitude = rng.uniform(0.25, 0.7)
            workload += amplitude * np.exp(-((t - center) ** 2) / (2 * width**2))
        workload = scale * workload + rng.normal(0, 0.025, size=cfg.n_steps)
        workload = np.clip(workload, 0.0, None)

        for metric_idx in range(cfg.n_metrics):
            lag = min(metric_idx * 2, cfg.n_steps - 1)
            lagged = np.roll(workload, lag)
            metric_noise = rng.normal(0, 0.02 + 0.01 * metric_idx, size=cfg.n_steps)
            cross = 0.04 * metric_idx * daily + 0.02 * metric_idx * weekly
            series[entity_idx, :, metric_idx] = np.clip(lagged * (1 + 0.12 * metric_idx) + cross + metric_noise, 0, None)

    metric_names = ["cpu", "memory", "request_count", "network_in"][: cfg.n_metrics]
    if len(metric_names) < cfg.n_metrics:
        metric_names.extend([f"metric_{i}" for i in range(len(metric_names), cfg.n_metrics)])
    entity_ids = [f"svc_{i:03d}" for i in range(cfg.n_entities)]
    return series, metric_names, entity_ids


def load_csv_cloudops(cfg: DataConfig) -> tuple[np.ndarray, list[str], list[str]]:
    if cfg.csv_path is None:
        raise ValueError("data.csv_path is required when data.source=csv")
    path = Path(cfg.csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = [cfg.timestamp_col, cfg.entity_col]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metric_cols = cfg.metric_cols
    if metric_cols is None:
        metric_cols = [c for c in frame.columns if c not in required]
    if not metric_cols:
        raise ValueError("No metric columns found")

    frame[cfg.timestamp_col] = pd.to_datetime(frame[cfg.timestamp_col])
    frame = frame.sort_values([cfg.entity_col, cfg.timestamp_col])
    entities = [str(v) for v in frame[cfg.entity_col].drop_duplicates().tolist()]
    all_times = pd.Index(sorted(frame[cfg.timestamp_col].drop_duplicates()))

    arrays = []
    for entity_id in entities:
        entity_frame = frame[frame[cfg.entity_col].astype(str) == entity_id]
        entity_frame = entity_frame.set_index(cfg.timestamp_col).reindex(all_times)
        values = entity_frame[metric_cols].interpolate(limit_direction="both").ffill().bfill()
        arrays.append(values.to_numpy(dtype=np.float32))
    return np.stack(arrays, axis=0), metric_cols, entities


def load_npz_cloudops(cfg: DataConfig) -> tuple[np.ndarray, list[str], list[str]]:
    if cfg.npz_path is None:
        raise ValueError("data.npz_path is required when data.source=npz")
    path = Path(cfg.npz_path)
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    if "series" not in data:
        raise ValueError("NPZ file must contain a 'series' array")
    series = data["series"].astype(np.float32)
    if series.ndim != 3:
        raise ValueError("NPZ 'series' must have shape [entities, time, metrics]")
    metric_names = data["metric_names"].astype(str).tolist() if "metric_names" in data else [f"metric_{i}" for i in range(series.shape[-1])]
    entity_ids = data["entity_ids"].astype(str).tolist() if "entity_ids" in data else [f"entity_{i}" for i in range(series.shape[0])]
    return series, metric_names, entity_ids


def normalize_by_train(series: np.ndarray, train_end: int) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_values = series[:, :train_end, :].reshape(-1, series.shape[-1])
    scaler.fit(train_values)
    flat = series.reshape(-1, series.shape[-1])
    normalized = scaler.transform(flat).reshape(series.shape).astype(np.float32)
    return normalized, scaler


def build_window_bundle(cfg: DataConfig, seed: int) -> WindowBundle:
    if cfg.source == "synthetic":
        series, metric_names, entity_ids = make_synthetic_cloudops(cfg, seed)
    elif cfg.source == "csv":
        series, metric_names, entity_ids = load_csv_cloudops(cfg)
    elif cfg.source == "npz":
        series, metric_names, entity_ids = load_npz_cloudops(cfg)
    else:
        raise ValueError(f"Unsupported data source: {cfg.source}")

    n_steps = series.shape[1]
    train_end = int(n_steps * cfg.train_ratio)
    val_end = int(n_steps * (cfg.train_ratio + cfg.val_ratio))
    normalized, scaler = normalize_by_train(series, train_end)

    train = WindowDataset(normalized, cfg.input_len, cfg.pred_len, cfg.target_metric, 0, train_end)
    val = WindowDataset(normalized, cfg.input_len, cfg.pred_len, cfg.target_metric, train_end - cfg.input_len, val_end)
    test = WindowDataset(normalized, cfg.input_len, cfg.pred_len, cfg.target_metric, val_end - cfg.input_len, n_steps)
    return WindowBundle(train=train, val=val, test_clean=test, scaler=scaler, metric_names=metric_names, entity_ids=entity_ids)
