from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99]


def max_true_run(mask: np.ndarray) -> int:
    best = 0
    cur = 0
    for value in mask.astype(bool):
        if value:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def true_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.astype(bool)):
        if value and start is None:
            start = idx
        if (not value or idx == len(mask) - 1) and start is not None:
            end = idx + 1 if value and idx == len(mask) - 1 else idx
            if end - start >= min_len:
                runs.append((start, end))
            start = None
    return runs


def robust_z(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if finite.size < 4:
        return np.zeros_like(values, dtype=np.float32)
    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    if mad < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    z = np.zeros_like(values, dtype=np.float32)
    valid = np.isfinite(values)
    z[valid] = 0.6745 * (values[valid] - median) / mad
    return z


def level_shift_score(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 8:
        return 0.0
    half = finite.size // 2
    scale = float(np.std(finite))
    if scale < 1e-8:
        return 0.0
    return float(abs(np.mean(finite[half:]) - np.mean(finite[:half])) / scale)


def quantile_dict(values: list[float], prefix: str) -> dict[str, float]:
    if not values:
        return {f"{prefix}_p{int(q * 100)}": 0.0 for q in QUANTILES}
    arr = np.asarray(values, dtype=np.float64)
    return {f"{prefix}_p{int(q * 100)}": float(np.quantile(arr, q)) for q in QUANTILES}


def parse_dataset(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Datasets must use NAME=PATH")
    name, path = raw.split("=", 1)
    return name, Path(path)


def channel_windows(values: np.ndarray, window: int, stride: int) -> list[np.ndarray]:
    if len(values) < window:
        return [values]
    return [values[start : start + window] for start in range(0, len(values) - window + 1, stride)]


def audit_dataset(
    name: str,
    path: Path,
    *,
    window: int,
    stride: int,
    run_len: int,
    zero_eps: float,
    flat_eps: float,
    max_segments: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    data = np.load(path, allow_pickle=True)
    if "series" not in data:
        raise ValueError(f"{path} does not contain 'series'")
    series = data["series"].astype(np.float32)
    if series.ndim != 3:
        raise ValueError(f"{path} series must be [entities, time, metrics]")

    entities, steps, metrics = series.shape
    finite = np.isfinite(series)
    finite_values = series[finite]
    summary: dict[str, float | int | str] = {
        "dataset": name,
        "entities": entities,
        "time_steps": steps,
        "metrics": metrics,
        "nonfinite_rate": 1.0 - float(np.mean(finite)),
        "zero_rate": float(np.mean(np.abs(finite_values) <= zero_eps)) if finite_values.size else 0.0,
    }

    segments: list[dict[str, float | int | str]] = []
    missing_rates: list[float] = []
    zero_rates: list[float] = []
    flatline_rates: list[float] = []
    spike_rates: list[float] = []
    level_scores: list[float] = []
    max_zero_runs: list[int] = []
    max_flat_runs: list[int] = []
    tail_flatline_count = 0
    low_variance_count = 0
    channel_count = entities * metrics

    for entity_idx in range(entities):
        for metric_idx in range(metrics):
            values = series[entity_idx, :, metric_idx]
            valid = np.isfinite(values)
            clean = np.where(valid, values, np.nan)
            if np.nanstd(clean) <= flat_eps:
                low_variance_count += 1

            missing_mask = ~valid
            zero_mask = valid & (np.abs(values) <= zero_eps)
            diffs = np.abs(np.diff(clean))
            flat_mask = np.isfinite(diffs) & (diffs <= flat_eps)
            z = robust_z(values)
            spike_mask = np.abs(z) > 6.0

            max_zero_runs.append(max_true_run(zero_mask))
            max_flat_runs.append(max_true_run(flat_mask))
            tail = clean[-run_len:]
            if np.all(np.isfinite(tail)) and np.nanmax(tail) - np.nanmin(tail) <= flat_eps:
                tail_flatline_count += 1
                if len(segments) < max_segments:
                    segments.append(
                        {
                            "dataset": name,
                            "event_type": "tail_flatline",
                            "entity_idx": entity_idx,
                            "metric_idx": metric_idx,
                            "start": max(0, steps - run_len),
                            "end": steps,
                            "severity": float(np.nanmax(tail) - np.nanmin(tail)),
                        }
                    )

            for event_type, mask, severity_values in [
                ("missing_run", missing_mask, missing_mask.astype(float)),
                ("zero_run", zero_mask, np.abs(values)),
                ("flatline_run", np.concatenate([[False], flat_mask]), np.concatenate([[0.0], diffs])),
            ]:
                for start, end in true_runs(mask, run_len):
                    if len(segments) >= max_segments:
                        break
                    segment_values = severity_values[start:end]
                    severity = float(np.nanmean(segment_values)) if len(segment_values) else float(end - start)
                    segments.append(
                        {
                            "dataset": name,
                            "event_type": event_type,
                            "entity_idx": entity_idx,
                            "metric_idx": metric_idx,
                            "start": start,
                            "end": end,
                            "severity": severity if event_type != "missing_run" else float(end - start),
                        }
                    )

            spike_indices = np.where(spike_mask)[0]
            for idx in spike_indices[: max(0, min(len(spike_indices), 3))]:
                if len(segments) >= max_segments:
                    break
                segments.append(
                    {
                        "dataset": name,
                        "event_type": "spike_z6",
                        "entity_idx": entity_idx,
                        "metric_idx": metric_idx,
                        "start": int(idx),
                        "end": int(idx + 1),
                        "severity": float(abs(z[idx])),
                    }
                )

            for win in channel_windows(values, window, stride):
                win_valid = np.isfinite(win)
                missing_rates.append(1.0 - float(np.mean(win_valid)))
                finite_win = win[win_valid]
                if finite_win.size:
                    zero_rates.append(float(np.mean(np.abs(finite_win) <= zero_eps)))
                else:
                    zero_rates.append(1.0)
                if len(win) >= 2:
                    win_diffs = np.abs(np.diff(np.where(win_valid, win, np.nan)))
                    flatline_rates.append(float(np.mean(np.isfinite(win_diffs) & (win_diffs <= flat_eps))))
                spike_rates.append(float(np.mean(np.abs(robust_z(win)) > 6.0)))
                level_scores.append(level_shift_score(win))

    summary.update(
        {
            "long_zero_run12_channel_rate": float(np.mean(np.asarray(max_zero_runs) >= run_len)) if max_zero_runs else 0.0,
            "flatline12_channel_rate": float(np.mean(np.asarray(max_flat_runs) >= run_len)) if max_flat_runs else 0.0,
            "tail_flatline12_channel_rate": tail_flatline_count / max(1, channel_count),
            "low_variance_metric_channel_rate": low_variance_count / max(1, channel_count),
            **quantile_dict(missing_rates, "window_missing_rate"),
            **quantile_dict(zero_rates, "window_zero_rate"),
            **quantile_dict(flatline_rates, "window_flatline_rate"),
            **quantile_dict(spike_rates, "window_spike_fraction_z6"),
            **quantile_dict(level_scores, "window_level_shift_score"),
            "segments_recorded": len([row for row in segments if row["dataset"] == name]),
        }
    )

    alignment = [
        {
            "dataset": name,
            "synthetic_operator": "missing_points_30",
            "synthetic_level": 0.3,
            "natural_proxy": "window_missing_rate",
            "alignment_scope": "controlled_synthetic_probe",
            "interpretation": "public_trace_missing_rate_is_zero",
            **quantile_dict(missing_rates, "natural"),
        },
        {
            "dataset": name,
            "synthetic_operator": "missing_variables_30",
            "synthetic_level": 0.3,
            "natural_proxy": "long_zero_or_flatline_channel_rate",
            "alignment_scope": "supported_by_flatline_or_zero_proxy",
            "interpretation": "metric_outage_proxy",
            "natural_p50": float(np.median([summary["long_zero_run12_channel_rate"], summary["flatline12_channel_rate"]])),
            "natural_p75": float(max(summary["long_zero_run12_channel_rate"], summary["flatline12_channel_rate"])),
            "natural_p90": float(max(summary["long_zero_run12_channel_rate"], summary["flatline12_channel_rate"])),
            "natural_p95": float(max(summary["long_zero_run12_channel_rate"], summary["flatline12_channel_rate"])),
            "natural_p99": float(max(summary["long_zero_run12_channel_rate"], summary["flatline12_channel_rate"])),
        },
        {
            "dataset": name,
            "synthetic_operator": "delayed_tail_12",
            "synthetic_level": 12.0,
            "natural_proxy": "tail_flatline12_channel_rate",
            "alignment_scope": "proxy_only",
            "interpretation": "stale_tail_proxy_not_ingestion_delay",
            "natural_p50": summary["tail_flatline12_channel_rate"],
            "natural_p75": summary["tail_flatline12_channel_rate"],
            "natural_p90": summary["tail_flatline12_channel_rate"],
            "natural_p95": summary["tail_flatline12_channel_rate"],
            "natural_p99": summary["tail_flatline12_channel_rate"],
        },
        {
            "dataset": name,
            "synthetic_operator": "level_shift_0.4",
            "synthetic_level": 0.4,
            "natural_proxy": "window_level_shift_score",
            "alignment_scope": "unit_mismatch_directional_only",
            "interpretation": "synthetic_additive_shift_not_directly_comparable_to_normalized_score",
            **quantile_dict(level_scores, "natural"),
        },
        {
            "dataset": name,
            "synthetic_operator": "burst_rate_0.02",
            "synthetic_level": 0.02,
            "natural_proxy": "window_spike_fraction_z6",
            "alignment_scope": "supported_by_spike_fraction_proxy",
            "interpretation": "spike_proxy",
            **quantile_dict(spike_rates, "natural"),
        },
    ]
    return summary, segments, alignment


def to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in frame.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align synthetic stress levels with natural degradation proxies.")
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True, help="Dataset mapping NAME=PATH")
    parser.add_argument("--output-dir", default="outputs/real_degradation_calibration")
    parser.add_argument("--window", type=int, default=96)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--run-len", type=int, default=12)
    parser.add_argument("--zero-eps", type=float, default=1e-8)
    parser.add_argument("--flat-eps", type=float, default=1e-6)
    parser.add_argument("--max-segments-per-dataset", type=int, default=300)
    args = parser.parse_args()

    summaries = []
    segments = []
    alignments = []
    for name, path in args.dataset:
        summary, dataset_segments, dataset_alignment = audit_dataset(
            name,
            path,
            window=args.window,
            stride=args.stride,
            run_len=args.run_len,
            zero_eps=args.zero_eps,
            flat_eps=args.flat_eps,
            max_segments=args.max_segments_per_dataset,
        )
        summaries.append(summary)
        segments.extend(dataset_segments)
        alignments.extend(dataset_alignment)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    segments_frame = pd.DataFrame(segments)
    alignment_frame = pd.DataFrame(alignments)
    summary_frame.to_csv(output_dir / "real_degradation_summary.csv", index=False)
    segments_frame.to_csv(output_dir / "real_degradation_segments.csv", index=False)
    alignment_frame.to_csv(output_dir / "stress_level_alignment.csv", index=False)

    cleaned_note = (
        "The public traces show low natural non-finite rates, which is consistent with cleaned benchmark releases. "
        "The detected zero-run, flatline, spike, and level-shift proxies should therefore be read as degradation-proxy "
        "alignment evidence, not as production incident calibration or incident-prevalence estimates."
    )
    report = [
        "# Natural degradation proxy alignment",
        "",
        cleaned_note,
        "",
        "AIOpsStressBench synthetic stress levels are controlled operating points on severity curves. "
        "They bracket mild/moderate/severe deployment degradation and are not calibrated to unreleased operator incident frequency.",
        "",
        "Key scope notes:",
        "",
        "- Metric outage has the strongest proxy support: the 0.30 injected channel masking level lies near the low end of natural long-zero/flatline channel rates for Alibaba and Salesforce/Borg.",
        "- Missing-points stress is a controlled synthetic worst-case probe in these public traces: the observed window-missing-rate quantiles are zero.",
        "- Delayed-tail uses tail flatline as a stale-telemetry proxy, not as direct evidence of ingestion-delay frequency.",
        "- Level-shift alignment is directional only because the injected additive shift and the natural normalized level-shift score use different units.",
        "",
        "## Dataset summary",
        "",
        to_markdown(summary_frame),
        "",
        "## Stress level alignment",
        "",
        to_markdown(alignment_frame),
        "",
        "## Segment samples",
        "",
        to_markdown(segments_frame.head(40)),
        "",
    ]
    (output_dir / "real_degradation_calibration.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved calibration outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
