from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


METRIC_NAMES = np.array(["min_cpu", "max_cpu", "avg_cpu"])


def iter_rows(path: Path, partial_ok: bool):
    try:
        with gzip.open(path, "rt", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                if len(row) < 5:
                    continue
                try:
                    yield int(row[0]), row[1], float(row[2]), float(row[3]), float(row[4])
                except ValueError:
                    continue
    except EOFError:
        if not partial_ok:
            raise
        print(f"WARNING: reached an incomplete gzip footer in {path}; writing a partial dataset.")


def count_entities(path: Path, max_rows: int, partial_ok: bool) -> Counter[str]:
    counts: Counter[str] = Counter()
    for idx, (_ts, vm_id, _min_cpu, _max_cpu, _avg_cpu) in enumerate(iter_rows(path, partial_ok), start=1):
        counts[vm_id] += 1
        if max_rows > 0 and idx >= max_rows:
            break
    return counts


def collect_series(path: Path, entity_ids: set[str], max_rows: int, partial_ok: bool) -> dict[str, dict[int, tuple[float, float, float]]]:
    values: dict[str, dict[int, tuple[float, float, float]]] = defaultdict(dict)
    for idx, (ts, vm_id, min_cpu, max_cpu, avg_cpu) in enumerate(iter_rows(path, partial_ok), start=1):
        if vm_id in entity_ids:
            values[vm_id][ts] = (min_cpu, max_cpu, avg_cpu)
        if max_rows > 0 and idx >= max_rows:
            break
    return values


def build_tensor(
    values: dict[str, dict[int, tuple[float, float, float]]],
    ordered_entities: list[str],
    max_length: int,
    min_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    kept_entities = []
    tensors = []
    for entity_id in ordered_entities:
        per_ts = values.get(entity_id, {})
        if len(per_ts) < min_length:
            continue
        timestamps = sorted(per_ts)[-max_length:]
        arr = np.full((len(timestamps), len(METRIC_NAMES)), np.nan, dtype=np.float32)
        for row_idx, ts in enumerate(timestamps):
            arr[row_idx] = np.asarray(per_ts[ts], dtype=np.float32)
        finite = np.isfinite(arr).all(axis=1)
        arr = arr[finite]
        if arr.shape[0] < min_length:
            continue
        tensors.append(arr)
        kept_entities.append(entity_id)

    if not tensors:
        raise SystemExit("No entities satisfied the requested min-length threshold.")

    length = min(tensor.shape[0] for tensor in tensors)
    length = min(length, max_length)
    output = np.stack([tensor[-length:] for tensor in tensors], axis=0).astype(np.float32)
    return output, np.asarray(kept_entities)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Azure VM CPU readings to AIOpsStressBench NPZ format.")
    parser.add_argument("--input", default="data/raw/azure_v2/vm_cpu_readings-file-1-of-195.csv.gz")
    parser.add_argument("--output", default="data/azure_vm_cpu_partial.npz")
    parser.add_argument("--max-entities", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--min-length", type=int, default=1000)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional row cap for development runs; 0 means all readable rows.")
    parser.add_argument("--partial-ok", action="store_true", help="Allow incomplete gzip files and mark the output as partial.")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file does not exist: {input_path}")

    counts = count_entities(input_path, args.max_rows, args.partial_ok)
    candidates = [entity for entity, length in counts.most_common() if length >= args.min_length]
    ordered_entities = candidates[: args.max_entities]
    if not ordered_entities:
        raise SystemExit(f"No entities have at least {args.min_length} readable points.")

    values = collect_series(input_path, set(ordered_entities), args.max_rows, args.partial_ok)
    series, entity_ids = build_tensor(values, ordered_entities, args.max_length, args.min_length)

    metadata = {
        "source": "AzurePublicDataset VM CPU readings",
        "input": str(input_path),
        "partial": bool(args.partial_ok),
        "max_rows": int(args.max_rows),
        "caveat": "CPU-only VM workload trace; partial outputs must not be used as main paper evidence.",
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        series=series,
        metric_names=METRIC_NAMES,
        entity_ids=entity_ids,
        metadata=np.asarray([metadata], dtype=object),
    )
    print(f"Saved {series.shape} to {output.resolve()}")
    print(f"partial={metadata['partial']} max_rows={metadata['max_rows']} min_length={args.min_length}")


if __name__ == "__main__":
    main()
