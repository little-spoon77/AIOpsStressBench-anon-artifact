from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    source: str = "synthetic"
    csv_path: str | None = None
    npz_path: str | None = None
    timestamp_col: str = "timestamp"
    entity_col: str = "entity_id"
    metric_cols: list[str] | None = None
    n_entities: int = 16
    n_steps: int = 900
    n_metrics: int = 4
    freq_minutes: int = 5
    input_len: int = 96
    pred_len: int = 24
    train_ratio: float = 0.65
    val_ratio: float = 0.15
    target_metric: int = 0


@dataclass
class StressConfig:
    scenario: str = "clean"
    missing_rate: float = 0.0
    noise_std: float = 0.0
    delay_steps: int = 0
    burst_rate: float = 0.0
    level_shift: float = 0.0
    imputation: str = "none"


@dataclass
class TrainConfig:
    models: list[str] = field(default_factory=lambda: ["last_value", "dlinear"])
    epochs: int = 4
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 2
    num_workers: int = 0
    device: str = "auto"
    latency_warmup: int = 8
    latency_iters: int = 40
    train_stress: bool = True
    calibrate: bool = True


@dataclass
class CapacityConfig:
    headroom: float = 0.15
    under_cost: float = 5.0
    over_cost: float = 1.0
    demand_floor: float = 0.05


@dataclass
class ExperimentConfig:
    seed: int = 42
    output_dir: str = "outputs/quick_synthetic"
    data: DataConfig = field(default_factory=DataConfig)
    stress: StressConfig = field(default_factory=StressConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)


def _merge_dataclass(cls: type, values: dict[str, Any] | None):
    base = cls()
    if values is None:
        return base
    for key, value in values.items():
        if not hasattr(base, key):
            raise ValueError(f"Unknown config key for {cls.__name__}: {key}")
        setattr(base, key, value)
    return base


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = ExperimentConfig()
    for key, value in raw.items():
        if key == "data":
            cfg.data = _merge_dataclass(DataConfig, value)
        elif key == "stress":
            cfg.stress = _merge_dataclass(StressConfig, value)
        elif key == "train":
            cfg.train = _merge_dataclass(TrainConfig, value)
        elif key == "capacity":
            cfg.capacity = _merge_dataclass(CapacityConfig, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    return cfg
