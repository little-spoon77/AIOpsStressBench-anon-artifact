from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from race_forecast.config import StressConfig, load_config
from race_forecast.data import build_window_bundle
from race_forecast.train import select_device, train_one_model


SCENARIOS = {
    "clean": {"scenario": "clean"},
    "missing_10": {"scenario": "missing_points", "missing_rate": 0.1},
    "missing_30": {"scenario": "missing_points", "missing_rate": 0.3},
    "missing_50": {"scenario": "missing_points", "missing_rate": 0.5},
    "missing_70": {"scenario": "missing_points", "missing_rate": 0.7},
    "missing_variables_10": {"scenario": "missing_variables", "missing_rate": 0.1},
    "missing_variables_30": {"scenario": "missing_variables", "missing_rate": 0.3},
    "missing_variables_50": {"scenario": "missing_variables", "missing_rate": 0.5},
    "delayed_6": {"scenario": "delayed_tail", "delay_steps": 6},
    "delayed_12": {"scenario": "delayed_tail", "delay_steps": 12},
    "delayed_24": {"scenario": "delayed_tail", "delay_steps": 24},
    "noise_10": {"scenario": "noisy", "noise_std": 0.1},
    "noisy": {"scenario": "noisy", "noise_std": 0.2},
    "noise_40": {"scenario": "noisy", "noise_std": 0.4},
    "burst": {"scenario": "burst", "burst_rate": 0.02},
    "level_shift": {"scenario": "level_shift", "level_shift": 0.4},
}


class OfficialITransformerTarget(nn.Module):
    def __init__(
        self,
        itransformer_root: Path,
        input_len: int,
        pred_len: int,
        target_metric: int,
        d_model: int,
        d_ff: int,
        n_heads: int,
        e_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        root = str(itransformer_root.resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from model import iTransformer  # type: ignore

        args = SimpleNamespace(
            seq_len=input_len,
            pred_len=pred_len,
            output_attention=False,
            use_norm=True,
            d_model=d_model,
            embed="timeF",
            freq="h",
            dropout=dropout,
            class_strategy="projection",
            factor=1,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            activation="gelu",
            moving_avg=25,
            use_dasr=False,
        )
        self.model = iTransformer.Model(args)
        self.target_metric = target_metric

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        forecast = self.model(x, None, None, None)
        return forecast[:, :, self.target_metric]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_stress(values: dict[str, float | str]) -> StressConfig:
    cfg = StressConfig()
    for key, value in values.items():
        setattr(cfg, key, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official iTransformer inside the native RACE stress pipeline.")
    parser.add_argument("--itransformer-root", default="external/iTransformer_official_run")
    parser.add_argument("--base-config", default="configs/alibaba2018_machine_usage.yaml")
    parser.add_argument("--output-root", default="outputs/official_itransformer_native")
    parser.add_argument("--summary", default="outputs/official_itransformer_native_summary.csv")
    parser.add_argument("--source", default="alibaba2018")
    parser.add_argument("--dataset", default="alibaba2018")
    parser.add_argument("--scenarios", nargs="*", default=["clean", "missing_variables_30"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()

    itransformer_root = Path(args.itransformer_root)
    if not (itransformer_root / "model" / "iTransformer.py").exists():
        raise FileNotFoundError(f"Official iTransformer root is invalid: {itransformer_root}")

    rows = []
    for scenario in args.scenarios:
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        set_seed(args.seed)
        cfg = load_config(args.base_config)
        cfg.stress = make_stress(SCENARIOS[scenario])
        cfg.train.models = ["official_itransformer"]
        cfg.train.epochs = args.epochs
        cfg.train.batch_size = args.batch_size
        cfg.train.device = args.device
        output_dir = Path(args.output_root) / args.dataset / scenario
        device = select_device(cfg.train.device)
        bundle = build_window_bundle(cfg.data, args.seed)
        model = OfficialITransformerTarget(
            itransformer_root=itransformer_root,
            input_len=cfg.data.input_len,
            pred_len=cfg.data.pred_len,
            target_metric=cfg.data.target_metric,
            d_model=args.d_model,
            d_ff=args.d_ff,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            dropout=args.dropout,
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = train_one_model(
            model=model,
            name="official_itransformer",
            train_dataset=bundle.train,
            val_dataset=bundle.val,
            test_dataset=bundle.test_clean,
            train_cfg=cfg.train,
            stress_cfg=cfg.stress,
            capacity_cfg=cfg.capacity,
            output_dir=output_dir,
            device=device,
            target_mean=float(bundle.scaler.mean_[cfg.data.target_metric]),
            target_scale=float(bundle.scaler.scale_[cfg.data.target_metric]),
        )
        rows.append({"source": args.source, "dataset": args.dataset, "stress": scenario, **result})
        rows[-1]["model"] = "official_itransformer_native"
        print(pd.DataFrame(rows).to_string(index=False))

    output = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
