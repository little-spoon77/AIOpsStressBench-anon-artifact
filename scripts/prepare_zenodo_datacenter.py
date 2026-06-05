from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


ZENODO_RECORD = "14564935"
DATASETS = {
    "alibaba_machine_300s": {
        "file": "machine_usage_days_1_to_8_grouped_300_seconds.csv",
        "preferred_metrics": ["cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"],
    },
    "google_instance_300s": {
        "file": "instance_usage_grouped_300_seconds_month.csv",
        "preferred_metrics": ["avg_cpu", "avg_mem", "avg_assigned_mem", "avg_cycles_per_instruction"],
    },
    "azure_vm_300s": {
        "file": "vm_cpu_readings_month_aggregated_cpu_mem.csv",
        "preferred_metrics": ["cpu_usage", "assigned_mem"],
    },
}


def download(url: str, output: Path, retries: int, retry_sleep: float) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 0:
        print(f"Using existing file: {output}")
        return
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as response, output.open("wb") as f:
                f.write(response.read())
            return
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
            print(f"Download failed on attempt {attempt}/{retries}: {exc}. Retrying in {retry_sleep}s.")
            time.sleep(retry_sleep)
    raise RuntimeError("Download failed") from last_error


def dataframe_to_npz(frame: pd.DataFrame, preferred_metrics: list[str], output: Path, max_length: int) -> None:
    metrics = [col for col in preferred_metrics if col in frame.columns]
    if not metrics:
        metrics = frame.select_dtypes(include=["number"]).columns.tolist()
    if not metrics:
        raise ValueError("No numeric metric columns found")

    values = frame[metrics].replace([np.inf, -np.inf], np.nan)
    values = values.interpolate(limit_direction="both").ffill().bfill().to_numpy(dtype=np.float32)
    if max_length > 0:
        values = values[-max_length:]
    series = values.reshape(1, values.shape[0], values.shape[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        series=series,
        metric_names=np.asarray(metrics),
        entity_ids=np.asarray(["datacenter_aggregate"]),
    )
    print(f"Saved {series.shape} to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare processed Zenodo datacenter traces as RACE-Forecast NPZ.")
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="alibaba_machine_300s")
    parser.add_argument("--raw-file", default=None, help="Use an already downloaded CSV instead of downloading from Zenodo.")
    parser.add_argument("--raw-dir", default="data/raw/zenodo_datacenter")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=10.0)
    args = parser.parse_args()

    meta = DATASETS[args.dataset]
    filename = meta["file"]
    raw_path = Path(args.raw_file) if args.raw_file else Path(args.raw_dir) / filename
    output = Path(args.output) if args.output else Path("data") / f"{args.dataset}.npz"
    url = f"https://zenodo.org/records/{ZENODO_RECORD}/files/{filename}?download=1"

    if args.raw_file is None:
        download(url, raw_path, args.retries, args.retry_sleep)
    elif not raw_path.exists():
        raise FileNotFoundError(raw_path)
    frame = pd.read_csv(raw_path)
    dataframe_to_npz(frame, meta["preferred_metrics"], output, args.max_length)


if __name__ == "__main__":
    main()
