from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert a generic telemetry CSV to RACE-Forecast NPZ.")
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output NPZ path.")
    parser.add_argument("--metric-cols", nargs="+", required=True, help="Numeric metric columns.")
    parser.add_argument("--entity-col", default=None, help="Optional entity column. If omitted, the CSV is one entity.")
    parser.add_argument("--time-col", default=None, help="Optional time column used for sorting.")
    parser.add_argument("--max-entities", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=0)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    missing = [col for col in args.metric_cols if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")

    if args.time_col and args.time_col in frame.columns:
        try:
            frame[args.time_col] = pd.to_datetime(frame[args.time_col])
        except (TypeError, ValueError):
            pass
        sort_cols = [args.time_col]
    else:
        sort_cols = []

    arrays = []
    entity_ids = []
    if args.entity_col and args.entity_col in frame.columns:
        groups = list(frame.groupby(args.entity_col, sort=False))
        if args.max_entities > 0:
            groups = groups[: args.max_entities]
        min_len = min(len(group) for _, group in groups)
        if args.max_length > 0:
            min_len = min(min_len, args.max_length)
        for entity_id, group in groups:
            if sort_cols:
                group = group.sort_values(sort_cols)
            values = group[args.metric_cols].replace([np.inf, -np.inf], np.nan)
            values = values.interpolate(limit_direction="both").ffill().bfill()
            arrays.append(values.to_numpy(dtype=np.float32)[-min_len:])
            entity_ids.append(str(entity_id))
    else:
        if sort_cols:
            frame = frame.sort_values(sort_cols)
        values = frame[args.metric_cols].replace([np.inf, -np.inf], np.nan)
        values = values.interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
        if args.max_length > 0:
            values = values[-args.max_length :]
        arrays.append(values)
        entity_ids.append("entity_0")

    series = np.stack(arrays, axis=0)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        series=series,
        metric_names=np.asarray(args.metric_cols),
        entity_ids=np.asarray(entity_ids),
    )
    print(f"Saved {series.shape} to {output.resolve()}")


if __name__ == "__main__":
    main()
