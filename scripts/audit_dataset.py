from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def audit_npz(path: Path) -> None:
    data = np.load(path, allow_pickle=True)
    series = data["series"].astype(np.float32)
    metric_names = data["metric_names"].astype(str).tolist() if "metric_names" in data else [f"metric_{i}" for i in range(series.shape[-1])]
    finite = np.isfinite(series)
    zero_rate = np.mean(series == 0, axis=(0, 1))
    print(f"path: {path}")
    print(f"shape [entities,time,metrics]: {series.shape}")
    print(f"metric_names: {metric_names}")
    print(f"finite_rate: {float(finite.mean()):.6f}")
    print("per_metric:")
    for idx, name in enumerate(metric_names):
        values = series[:, :, idx]
        print(
            f"  {name}: mean={float(np.nanmean(values)):.6f}, std={float(np.nanstd(values)):.6f}, "
            f"min={float(np.nanmin(values)):.6f}, max={float(np.nanmax(values)):.6f}, zero_rate={float(zero_rate[idx]):.6f}"
        )


def audit_csv(path: Path, timestamp_col: str, entity_col: str, metric_cols: list[str] | None) -> None:
    frame = pd.read_csv(path)
    if metric_cols is None:
        metric_cols = [c for c in frame.columns if c not in {timestamp_col, entity_col}]
    print(f"path: {path}")
    print(f"rows: {len(frame)}")
    print(f"entities: {frame[entity_col].nunique()}")
    print(f"metric_cols: {metric_cols}")
    frame[timestamp_col] = pd.to_datetime(frame[timestamp_col])
    lengths = frame.groupby(entity_col)[timestamp_col].nunique()
    print(f"time_points_per_entity: min={int(lengths.min())}, median={float(lengths.median()):.1f}, max={int(lengths.max())}")
    print("per_metric:")
    for col in metric_cols:
        values = frame[col]
        print(
            f"  {col}: missing={float(values.isna().mean()):.6f}, mean={float(values.mean()):.6f}, "
            f"std={float(values.std()):.6f}, min={float(values.min()):.6f}, max={float(values.max()):.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a RACE-Forecast dataset.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--format", choices=["npz", "csv"], default="npz")
    parser.add_argument("--timestamp-col", default="timestamp")
    parser.add_argument("--entity-col", default="entity_id")
    parser.add_argument("--metric-cols", nargs="*", default=None)
    args = parser.parse_args()

    path = Path(args.path)
    if args.format == "npz":
        audit_npz(path)
    else:
        audit_csv(path, args.timestamp_col, args.entity_col, args.metric_cols)


if __name__ == "__main__":
    main()

