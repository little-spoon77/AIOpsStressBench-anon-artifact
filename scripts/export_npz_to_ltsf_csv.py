from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() else "_")
    name = "".join(keep).strip("_")
    return name or "series"


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RACE-Forecast NPZ data to LTSF CSV format.")
    parser.add_argument("--input", required=True, help="NPZ file with series [entities,time,metrics].")
    parser.add_argument("--output", required=True, help="CSV output path.")
    parser.add_argument("--max-entities", type=int, default=64)
    parser.add_argument("--max-metrics", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--freq", default="5min", help="Pandas frequency for synthetic date column.")
    parser.add_argument("--start", default="2020-01-01")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    series = data["series"].astype(np.float32)
    if series.ndim != 3:
        raise ValueError("NPZ series must have shape [entities,time,metrics]")
    metric_names = data["metric_names"].astype(str).tolist() if "metric_names" in data else [f"metric_{i}" for i in range(series.shape[2])]
    entity_ids = data["entity_ids"].astype(str).tolist() if "entity_ids" in data else [f"entity_{i}" for i in range(series.shape[0])]

    n_entities = min(series.shape[0], args.max_entities if args.max_entities > 0 else series.shape[0])
    n_metrics = min(series.shape[2], args.max_metrics if args.max_metrics > 0 else series.shape[2])
    n_steps = min(series.shape[1], args.max_length if args.max_length > 0 else series.shape[1])
    clipped = series[:n_entities, -n_steps:, :n_metrics]

    columns: dict[str, np.ndarray] = {}
    for entity_idx in range(n_entities):
        entity = safe_name(entity_ids[entity_idx])
        for metric_idx in range(n_metrics):
            metric = safe_name(metric_names[metric_idx])
            col = f"{entity}__{metric}"
            values = clipped[entity_idx, :, metric_idx]
            finite = np.isfinite(values)
            if not finite.all():
                values = pd.Series(values).interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
            columns[col] = values

    frame = pd.DataFrame(columns)
    frame.insert(0, "date", pd.date_range(args.start, periods=n_steps, freq=args.freq))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    print(f"Saved LTSF CSV: {output.resolve()}")
    print(f"shape: {frame.shape}")
    print(f"enc_in/dec_in/c_out for official baselines: {frame.shape[1] - 1}")
    print("Example iTransformer command:")
    print(
        "python run.py --is_training 1 --root_path "
        f"{output.parent.resolve()} --data_path {output.name} --model_id {output.stem}_96_24 "
        "--model iTransformer --data custom --features M --seq_len 96 --label_len 48 --pred_len 24 "
        f"--enc_in {frame.shape[1] - 1} --dec_in {frame.shape[1] - 1} --c_out {frame.shape[1] - 1} "
        "--d_model 128 --d_ff 256 --e_layers 2 --d_layers 1 --batch_size 32 --train_epochs 10"
    )


if __name__ == "__main__":
    main()
