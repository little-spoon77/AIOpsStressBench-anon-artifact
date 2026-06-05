from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GAIA metric_forecast.zip to RACE-Forecast NPZ.")
    parser.add_argument("--zip", default="data/raw/gaia/metric_forecast.zip")
    parser.add_argument("--output", default="data/gaia_metric_forecast.npz")
    parser.add_argument("--categories", nargs="*", default=None, help="Optional category directory filters.")
    parser.add_argument("--max-series", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    category_filters = set(args.categories or [])

    arrays = []
    entity_ids = []
    with zipfile.ZipFile(zip_path) as z:
        names = [name for name in z.namelist() if name.endswith(".csv")]
        names = sorted(names)
        for name in names:
            parts = Path(name).parts
            category = parts[1] if len(parts) >= 3 else "unknown"
            if category_filters and category not in category_filters:
                continue
            with z.open(name) as f:
                frame = pd.read_csv(f)
            if "value" not in frame.columns:
                continue
            values = frame["value"].replace([np.inf, -np.inf], np.nan)
            values = values.interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
            if args.max_length > 0:
                values = values[-args.max_length :]
            if len(values) < 128:
                continue
            arrays.append(values.reshape(-1, 1))
            entity_ids.append(f"{category}/{Path(name).stem}")
            if args.max_series > 0 and len(arrays) >= args.max_series:
                break

    if not arrays:
        raise SystemExit("No GAIA metric series were converted.")
    min_len = min(arr.shape[0] for arr in arrays)
    series = np.stack([arr[-min_len:] for arr in arrays], axis=0).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        series=series,
        metric_names=np.asarray(["value"]),
        entity_ids=np.asarray(entity_ids),
    )
    print(f"Saved {series.shape} to {output.resolve()}")


if __name__ == "__main__":
    main()

