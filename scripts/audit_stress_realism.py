from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def max_true_run(mask: np.ndarray) -> int:
    if mask.size == 0:
        return 0
    best = 0
    current = 0
    for value in mask.astype(bool):
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def robust_spike_fraction(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return 0.0
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    if mad < 1e-8:
        return 0.0
    z = 0.6745 * (finite - median) / mad
    return float(np.mean(np.abs(z) > 6.0))


def level_shift_score(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 8:
        return 0.0
    half = finite.size // 2
    scale = float(np.std(finite))
    if scale < 1e-8:
        return 0.0
    return float(abs(np.mean(finite[half:]) - np.mean(finite[:half])) / scale)


def audit_dataset(name: str, path: Path, zero_eps: float, flat_eps: float, run_len: int) -> dict[str, float | int | str]:
    data = np.load(path, allow_pickle=True)
    if "series" not in data:
        raise ValueError(f"{path} does not contain a 'series' array")
    series = data["series"].astype(np.float32)
    if series.ndim != 3:
        raise ValueError(f"{path} series must have shape [entities, time, metrics]")

    entities, steps, metrics = series.shape
    finite = np.isfinite(series)
    finite_values = series[finite]
    nonfinite_rate = 1.0 - float(np.mean(finite))
    zero_rate = float(np.mean(np.abs(finite_values) <= zero_eps)) if finite_values.size else 0.0

    zero_run_channels = 0
    flatline_channels = 0
    tail_flatline_channels = 0
    spike_fractions = []
    shift_scores = []
    metric_stds = []

    for metric_idx in range(metrics):
        metric_values = series[:, :, metric_idx]
        metric_stds.append(float(np.nanstd(metric_values)))
        for entity_idx in range(entities):
            values = metric_values[entity_idx]
            valid = np.isfinite(values)
            clean = np.where(valid, values, np.nan)
            zero_mask = valid & (np.abs(values) <= zero_eps)
            if max_true_run(zero_mask) >= run_len:
                zero_run_channels += 1

            diffs = np.abs(np.diff(clean))
            flat_mask = np.isfinite(diffs) & (diffs <= flat_eps)
            if max_true_run(flat_mask) >= max(1, run_len - 1):
                flatline_channels += 1

            tail = clean[-run_len:]
            if np.all(np.isfinite(tail)) and (np.nanmax(tail) - np.nanmin(tail) <= flat_eps):
                tail_flatline_channels += 1

            spike_fractions.append(robust_spike_fraction(values))
            shift_scores.append(level_shift_score(values))

    channel_count = max(1, entities * metrics)
    low_variance_metric_rate = float(np.mean(np.asarray(metric_stds) <= flat_eps)) if metric_stds else 0.0
    return {
        "dataset": name,
        "entities": entities,
        "time_steps": steps,
        "metrics": metrics,
        "nonfinite_rate": nonfinite_rate,
        "zero_rate": zero_rate,
        "long_zero_run12_channel_rate": zero_run_channels / channel_count,
        "flatline12_channel_rate": flatline_channels / channel_count,
        "tail_flatline12_channel_rate": tail_flatline_channels / channel_count,
        "low_variance_metric_rate": low_variance_metric_rate,
        "spike_fraction_z6": float(np.mean(spike_fractions)) if spike_fractions else 0.0,
        "median_level_shift_score": float(np.median(shift_scores)) if shift_scores else 0.0,
        "p95_level_shift_score": float(np.percentile(shift_scores, 95)) if shift_scores else 0.0,
    }


def to_markdown(frame: pd.DataFrame) -> str:
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def parse_dataset(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Datasets must use NAME=PATH")
    name, path = raw.split("=", 1)
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit natural telemetry degradation signals in public operational traces.")
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True, help="Dataset mapping NAME=PATH")
    parser.add_argument("--output-dir", default="outputs/paper_tables")
    parser.add_argument("--zero-eps", type=float, default=1e-8)
    parser.add_argument("--flat-eps", type=float, default=1e-6)
    parser.add_argument("--run-len", type=int, default=12)
    args = parser.parse_args()

    rows = [audit_dataset(name, path, args.zero_eps, args.flat_eps, args.run_len) for name, path in args.dataset]
    frame = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_stress_realism_audit.csv"
    md_path = output_dir / "table_stress_realism_audit.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(frame) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
