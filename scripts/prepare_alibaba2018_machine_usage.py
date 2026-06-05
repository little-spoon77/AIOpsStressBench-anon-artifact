from __future__ import annotations

import argparse
import csv
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


URL_BASE = "http://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
USAGE_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mkpi",
    "net_in",
    "net_out",
    "disk_io_percent",
]


def extract_first_csv(archive_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".csv")]
        if not members:
            raise ValueError(f"No CSV file found in {archive_path}")
        member = members[0]
        output_path = output_dir / Path(member.name).name
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path
        extracted = archive.extractfile(member)
        if extracted is None:
            raise ValueError(f"Cannot extract {member.name}")
        with output_path.open("wb") as target:
            while True:
                chunk = extracted.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
        return output_path


def infer_header(path: Path) -> tuple[bool, list[str]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        first = f.readline().strip().split(",")
    if first and first[0] == "machine_id":
        return True, first
    return False, USAGE_COLUMNS


def csv_to_npz(
    csv_path: Path,
    output_path: Path,
    max_machines: int,
    max_length: int,
    metrics: list[str],
) -> None:
    by_machine: dict[str, list[tuple[int, list[float]]]] = defaultdict(list)
    selected: list[str] = []
    selected_set: set[str] = set()
    has_header, header = infer_header(csv_path)
    rows_seen = 0
    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, fieldnames=None if has_header else header)
        for row in reader:
            machine_id = row["machine_id"]
            if machine_id not in selected_set and (max_machines <= 0 or len(selected) < max_machines):
                selected.append(machine_id)
                selected_set.add(machine_id)
            if machine_id not in selected_set:
                continue
            if max_length > 0 and len(by_machine[machine_id]) >= max_length:
                if max_machines > 0 and len(selected) >= max_machines and all(len(by_machine[item]) >= max_length for item in selected):
                    break
                continue
            values = []
            for metric in metrics:
                raw = row.get(metric, "")
                try:
                    values.append(float(raw))
                except ValueError:
                    values.append(np.nan)
            try:
                timestamp = int(float(row["time_stamp"]))
            except ValueError:
                continue
            by_machine[machine_id].append((timestamp, values))
            rows_seen += 1
            if max_machines > 0 and max_length > 0 and len(selected) >= max_machines:
                if all(len(by_machine[item]) >= max_length for item in selected):
                    break

    if not selected:
        raise ValueError("No machine IDs found")

    arrays = []
    entity_ids = []
    for machine_id in selected:
        rows = sorted(by_machine[machine_id], key=lambda item: item[0])
        if not rows:
            continue
        values = np.asarray([v for _, v in rows], dtype=np.float32)
        frame = pd.DataFrame(values, columns=metrics).replace([np.inf, -np.inf], np.nan)
        frame = frame.interpolate(limit_direction="both").ffill().bfill()
        values = frame.to_numpy(dtype=np.float32)
        if max_length > 0:
            values = values[:max_length]
        arrays.append(values)
        entity_ids.append(machine_id)

    if not arrays:
        raise ValueError("No rows matched selected machines")
    min_len = min(arr.shape[0] for arr in arrays)
    if max_length > 0:
        min_len = min(min_len, max_length)
    series = np.stack([arr[:min_len] for arr in arrays], axis=0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        series=series,
        metric_names=np.asarray(metrics),
        entity_ids=np.asarray(entity_ids),
    )
    print(f"Read {rows_seen} matching rows")
    print(f"Saved {series.shape} to {output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Alibaba cluster-trace-v2018 machine_usage to NPZ.")
    parser.add_argument("--archive", default="data/raw/alibaba2018/machine_usage.tar.gz")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--extract-dir", default="data/raw/alibaba2018/extracted")
    parser.add_argument("--output", default="data/alibaba2018_machine_usage.npz")
    parser.add_argument("--max-machines", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["cpu_util_percent", "mem_util_percent", "net_in", "net_out", "disk_io_percent"],
    )
    args = parser.parse_args()

    csv_path = Path(args.csv) if args.csv else extract_first_csv(Path(args.archive), Path(args.extract_dir))
    csv_to_npz(
        csv_path=csv_path,
        output_path=Path(args.output),
        max_machines=args.max_machines,
        max_length=args.max_length,
        metrics=args.metrics,
    )


if __name__ == "__main__":
    main()
