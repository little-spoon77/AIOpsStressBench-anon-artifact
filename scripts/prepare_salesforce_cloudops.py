from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


def _to_2d_dynamic(value, target_len: int) -> np.ndarray:
    if value is None:
        return np.zeros((0, target_len), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((0, target_len), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[-1] < target_len:
        return np.zeros((0, target_len), dtype=np.float32)
    return arr[..., -target_len:]


def _last_axis_length(value) -> int:
    arr = np.asarray(value)
    if arr.size == 0:
        return 0
    return int(arr.shape[-1])


def audit_series(series: np.ndarray, metric_names: np.ndarray, output: Path) -> None:
    finite = np.isfinite(series)
    rows = []
    for metric_idx, metric_name in enumerate(metric_names.astype(str).tolist()):
        values = series[:, :, metric_idx]
        rows.append(
            {
                "metric": metric_name,
                "missing_rate": float(1.0 - np.isfinite(values).mean()),
                "zero_rate": float(np.mean(values[np.isfinite(values)] == 0.0)) if np.isfinite(values).any() else np.nan,
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    summary = pd.DataFrame(rows)
    summary.insert(0, "entities", series.shape[0])
    summary.insert(1, "time_steps", series.shape[1])
    summary.insert(2, "metrics", series.shape[2])
    summary.insert(3, "finite_rate", float(finite.mean()))
    output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output, index=False)
    print(f"Saved audit to {output.resolve()}")


def load_rows_from_zip(zip_file: Path, max_items: int) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit("Install optional dependency: pyarrow") from exc

    rows: list[dict] = []
    with ZipFile(zip_file) as archive:
        names = sorted(name for name in archive.namelist() if name.endswith(".parquet"))
        if not names:
            raise SystemExit(f"No parquet files found in {zip_file}")
        for name in names:
            with archive.open(name) as handle:
                table = pq.read_table(handle, columns=["item_id", "target", "past_feat_dynamic_real"])
            rows.extend(table.to_pylist())
            if max_items > 0 and len(rows) >= max_items:
                return rows[:max_items]
    return rows


def impute_nonfinite(series: np.ndarray) -> tuple[np.ndarray, float]:
    raw_nonfinite_rate = float(1.0 - np.isfinite(series).mean())
    if raw_nonfinite_rate == 0.0:
        return series, raw_nonfinite_rate

    output = series.copy()
    for entity_idx in range(output.shape[0]):
        for metric_idx in range(output.shape[2]):
            values = pd.Series(output[entity_idx, :, metric_idx])
            values = values.replace([np.inf, -np.inf], np.nan)
            values = values.interpolate(limit_direction="both").ffill().bfill().fillna(0.0)
            output[entity_idx, :, metric_idx] = values.to_numpy(dtype=np.float32)
    return output.astype(np.float32), raw_nonfinite_rate


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Salesforce CloudOps HF data to RACE-Forecast NPZ.")
    parser.add_argument("--config", default="azure_vm_traces_2017", choices=[
        "azure_vm_traces_2017",
        "borg_cluster_data_2011",
        "alibaba_cluster_trace_2018",
    ])
    parser.add_argument("--split", default="train_test")
    parser.add_argument("--max-items", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=2048, help="Keep only the last N time steps per item.")
    parser.add_argument("--output", default="data/salesforce_azure_256.npz")
    parser.add_argument("--audit-output", default=None, help="Optional dataset audit CSV path.")
    parser.add_argument("--no-streaming", action="store_true", help="Disable HuggingFace streaming mode.")
    parser.add_argument("--cache-dir", default=None, help="Optional HuggingFace cache directory.")
    parser.add_argument("--load-from-disk", default=None, help="Optional local HF dataset directory saved by datasets.save_to_disk.")
    parser.add_argument("--zip-file", default=None, help="Optional local Salesforce CloudOps train_test/pretrain zip file.")
    parser.add_argument("--no-impute", action="store_true", help="Keep original non-finite values instead of interpolating them.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count for Hub connection timeouts.")
    parser.add_argument("--retry-sleep", type=float, default=10.0, help="Seconds between retries.")
    args = parser.parse_args()

    if args.zip_file:
        rows = load_rows_from_zip(Path(args.zip_file), args.max_items)
        streaming = False
    elif args.load_from_disk:
        try:
            from datasets import load_from_disk
        except ImportError as exc:
            raise SystemExit("Install optional dependency: datasets") from exc

        from datasets import load_from_disk

        loaded = load_from_disk(args.load_from_disk)
        dataset = loaded[args.split] if hasattr(loaded, "keys") and args.split in loaded.keys() else loaded
        streaming = False
        n_items = min(args.max_items, len(dataset)) if args.max_items > 0 else len(dataset)
        rows = [dataset[i] for i in range(n_items)]
    else:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise SystemExit("Install optional dependencies: datasets==2.12.0 fsspec==2023.5.0 gluonts") from exc

        streaming = not args.no_streaming
        last_error = None
        for attempt in range(1, args.retries + 1):
            try:
                dataset = load_dataset(
                    "Salesforce/cloudops_tsf",
                    args.config,
                    split=args.split,
                    streaming=streaming,
                    cache_dir=args.cache_dir,
                )
                break
            except TypeError:
                dataset = load_dataset(
                    "Salesforce/cloudops_tsf",
                    args.config,
                    split=args.split,
                    cache_dir=args.cache_dir,
                )
                break
            except Exception as exc:  # HuggingFace wraps network failures in several exception types.
                last_error = exc
                if attempt >= args.retries:
                    raise
                print(f"Dataset load failed on attempt {attempt}/{args.retries}: {exc}. Retrying in {args.retry_sleep}s.")
                time.sleep(args.retry_sleep)
        else:
            raise RuntimeError("Dataset load failed") from last_error

        if streaming:
            rows = list(itertools.islice(dataset, args.max_items))
        else:
            n_items = min(args.max_items, len(dataset)) if args.max_items > 0 else len(dataset)
            rows = [dataset[i] for i in range(n_items)]
    if not rows:
        raise SystemExit("No rows were loaded from the dataset.")

    target_lengths = [_last_axis_length(row["target"]) for row in rows]
    target_len = min(target_lengths)
    if args.max_length > 0:
        target_len = min(target_len, args.max_length)

    series = []
    entity_ids = []
    max_target = 0
    max_dyn = 0
    for row in rows:
        target = _to_2d_dynamic(row["target"], target_len)
        dynamic = _to_2d_dynamic(row.get("past_feat_dynamic_real"), target_len)
        max_target = max(max_target, target.shape[0])
        max_dyn = max(max_dyn, dynamic.shape[0])
        series.append((target, dynamic))
        entity_ids.append(str(row.get("item_id", len(entity_ids))))

    tensors = []
    for target, dynamic in series:
        if target.shape[0] < max_target:
            pad = np.zeros((max_target - target.shape[0], target_len), dtype=np.float32)
            target = np.concatenate([target, pad], axis=0)
        if dynamic.shape[0] < max_dyn:
            pad = np.zeros((max_dyn - dynamic.shape[0], target_len), dtype=np.float32)
            dynamic = np.concatenate([dynamic, pad], axis=0)
        tensors.append(np.concatenate([target, dynamic], axis=0).T)

    output_series = np.stack(tensors, axis=0).astype(np.float32)
    raw_nonfinite_rate = float(1.0 - np.isfinite(output_series).mean())
    if not args.no_impute:
        output_series, raw_nonfinite_rate = impute_nonfinite(output_series)
    target_names = ["target"] if max_target == 1 else [f"target_{i}" for i in range(max_target)]
    metric_names = np.array(target_names + [f"dynamic_{i}" for i in range(max_dyn)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "Salesforce/cloudops_tsf",
        "config": args.config,
        "split": args.split,
        "streaming": bool(streaming),
        "load_from_disk": args.load_from_disk,
        "zip_file": args.zip_file,
        "max_items": int(args.max_items),
        "max_length": int(args.max_length),
        "imputed_nonfinite": not args.no_impute,
        "raw_nonfinite_rate": raw_nonfinite_rate,
        "role": "CloudOps workload/resource trace candidate",
        "caveat": "Use as a main dataset only if audit satisfies entities>=50, time_steps>=1000, metrics>=2.",
        "partial": False,
    }
    np.savez_compressed(
        output,
        series=output_series,
        metric_names=metric_names,
        entity_ids=np.asarray(entity_ids),
        metadata=np.asarray([metadata], dtype=object),
    )
    print(f"Saved {output_series.shape} to {output.resolve()}")
    audit_output = Path(args.audit_output) if args.audit_output else output.with_name(f"{output.stem}_audit.csv")
    audit_series(output_series, metric_names, audit_output)


if __name__ == "__main__":
    main()
