from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd


DEFAULT_URL = "https://raw.githubusercontent.com/NetManAIOps/KPI-Anomaly-Detection/master/Finals_dataset/phase2_train.csv.zip"


def copy_stream(source: BinaryIO, output: Path, chunk_size: int, total: int | None) -> None:
    written = 0
    with output.open("wb") as f:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            written += len(chunk)
            if total:
                pct = written / total * 100
                print(f"\rDownloading {output.name}: {written / 1_048_576:.1f} MiB / {total / 1_048_576:.1f} MiB ({pct:.1f}%)", end="")
            else:
                print(f"\rDownloading {output.name}: {written / 1_048_576:.1f} MiB", end="")
    print()


def download(url: str, output: Path, timeout: int, chunk_size: int) -> None:
    import urllib.request

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"Using existing file: {output}")
        return
    temp_output = output.with_suffix(output.suffix + ".tmp")
    request = urllib.request.Request(url, headers={"User-Agent": "hsc2-race-forecast"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length_header = response.headers.get("Content-Length")
        total = int(length_header) if length_header else None
        copy_stream(response, temp_output, chunk_size, total)
    temp_output.replace(output)


def load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            csv_names = [name for name in z.namelist() if name.endswith(".csv")]
            if not csv_names:
                raise ValueError("No CSV found in zip")
            with z.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert NetManAIOps KPI dataset to RACE-Forecast NPZ.")
    parser.add_argument("--zip", default="data/raw/netman/phase2_train.csv.zip")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", default="data/netman_kpi.npz")
    parser.add_argument("--max-series", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    args = parser.parse_args()

    data_path = Path(args.zip)
    download(args.url, data_path, timeout=args.download_timeout, chunk_size=args.chunk_size)
    frame = load_frame(data_path)

    id_col = "KPI ID" if "KPI ID" in frame.columns else "kpi_id"
    ts_col = "timestamp" if "timestamp" in frame.columns else None
    value_col = "value"
    if id_col not in frame.columns or value_col not in frame.columns:
        raise ValueError(f"Expected columns including KPI ID/kpi_id and value, got {frame.columns.tolist()}")

    arrays = []
    entity_ids = []
    groups = list(frame.groupby(id_col, sort=False))
    if args.max_series > 0:
        groups = groups[: args.max_series]
    min_len = min(len(group) for _, group in groups)
    if args.max_length > 0:
        min_len = min(min_len, args.max_length)

    for entity_id, group in groups:
        if ts_col:
            group = group.sort_values(ts_col)
        values = group[value_col].replace([np.inf, -np.inf], np.nan)
        values = values.interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
        arrays.append(values[-min_len:].reshape(-1, 1))
        entity_ids.append(str(entity_id))

    series = np.stack(arrays, axis=0).astype(np.float32)
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
