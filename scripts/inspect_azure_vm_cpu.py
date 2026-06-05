from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Azure VM CPU readings for forecasting suitability.")
    parser.add_argument("--input", default="data/raw/azure_v2/vm_cpu_readings-file-1-of-195.full.csv.gz")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all readable rows.")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"Input does not exist: {path}")

    counts: Counter[str] = Counter()
    timestamps: Counter[int] = Counter()
    first_ts = None
    last_ts = None
    rows = 0
    with gzip.open(path, "rt", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5:
                continue
            ts = int(row[0])
            vm_id = row[1]
            first_ts = ts if first_ts is None else first_ts
            last_ts = ts
            counts[vm_id] += 1
            timestamps[ts] += 1
            rows += 1
            if args.max_rows > 0 and rows >= args.max_rows:
                break

    top_counts = counts.most_common(args.top_k)
    eligible_128 = sum(1 for value in counts.values() if value >= 128)
    eligible_1000 = sum(1 for value in counts.values() if value >= 1000)
    print(f"rows: {rows}")
    print(f"unique_vms: {len(counts)}")
    print(f"unique_timestamps: {len(timestamps)}")
    print(f"first_ts: {first_ts}")
    print(f"last_ts: {last_ts}")
    print(f"eligible_entities_len>=128: {eligible_128}")
    print(f"eligible_entities_len>=1000: {eligible_1000}")
    print("top_entity_counts:")
    for vm_id, count in top_counts:
        print(f"  {vm_id},{count}")


if __name__ == "__main__":
    main()
