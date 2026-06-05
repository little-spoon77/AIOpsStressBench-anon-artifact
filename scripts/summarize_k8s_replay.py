import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def avg(rows: Iterable[Dict[str, str]], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return mean(values) if values else 0.0


def sd(rows: Iterable[Dict[str, str]], key: str) -> float:
    values = [float(r[key]) for r in rows]
    return stdev(values) if len(values) > 1 else 0.0


def grouped(rows: Iterable[Dict[str, str]], keys: List[str]) -> Dict[tuple, List[Dict[str, str]]]:
    out: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return dict(out)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_source(source: str, rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    summaries: List[Dict[str, object]] = []
    for (stress, policy), group in sorted(grouped(rows, ["stress", "policy"]).items()):
        summaries.append(
            {
                "source": source,
                "stress": stress,
                "policy": policy,
                "n": len(group),
                "control_cost_mean": f"{avg(group, 'total_control_cost'):.6f}",
                "control_cost_std": f"{sd(group, 'total_control_cost'):.6f}",
                "p95_ms_mean": f"{avg(group, 'request_p95_latency_ms'):.3f}",
                "p95_ms_std": f"{sd(group, 'request_p95_latency_ms'):.3f}",
                "error_rate_mean": f"{avg(group, 'error_rate'):.6f}",
                "error_rate_std": f"{sd(group, 'error_rate'):.6f}",
                "overload_duration_mean": f"{avg(group, 'overload_duration'):.3f}",
                "peak_overload_mean": f"{avg(group, 'peak_overload_ratio'):.6f}",
                "replica_minutes_mean": f"{avg(group, 'replica_minutes'):.6f}",
                "scale_actions_mean": f"{avg(group, 'scale_action_count'):.3f}",
            }
        )
    return summaries


def objective_table(source: str, rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    by_stress: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(dict)
    for (stress, policy), group in grouped(rows, ["stress", "policy"]).items():
        by_stress[stress][policy] = group

    out: List[Dict[str, object]] = []
    for stress, by_policy in sorted(by_stress.items()):
        best_cost = min(by_policy, key=lambda p: avg(by_policy[p], "total_control_cost"))
        best_p95 = min(by_policy, key=lambda p: avg(by_policy[p], "request_p95_latency_ms"))
        max_error = max(avg(group, "error_rate") for group in by_policy.values())
        out.append(
            {
                "source": source,
                "stress": stress,
                "best_control_cost": best_cost,
                "control_cost_mean": f"{avg(by_policy[best_cost], 'total_control_cost'):.6f}",
                "best_p95_latency": best_p95,
                "p95_ms_mean": f"{avg(by_policy[best_p95], 'request_p95_latency_ms'):.3f}",
                "objective_disagreement": "Y" if best_cost != best_p95 else "N",
                "max_error_rate_mean": f"{max_error:.6f}",
            }
        )
    return out


def write_markdown(path: Path, summary_rows: List[Dict[str, object]], objective_rows: List[Dict[str, object]]) -> None:
    lines = ["# K8s Replay Record-and-Replay Summary", ""]
    lines.append("## Objective Disagreement")
    lines.append("")
    lines.append("| Source | Stress | Best cost | Cost | Best P95 | P95 ms | Disagree | Max error |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in objective_rows:
        lines.append(
            f"| {row['source']} | {row['stress']} | {row['best_control_cost']} | "
            f"{row['control_cost_mean']} | {row['best_p95_latency']} | {row['p95_ms_mean']} | "
            f"{row['objective_disagreement']} | {row['max_error_rate_mean']} |"
        )
    lines.append("")
    lines.append("## Mean +/- Std by Policy")
    lines.append("")
    lines.append("| Source | Stress | Policy | n | Cost mean | Cost std | P95 mean | P95 std | Error mean |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            f"| {row['source']} | {row['stress']} | {row['policy']} | {row['n']} | "
            f"{row['control_cost_mean']} | {row['control_cost_std']} | "
            f"{row['p95_ms_mean']} | {row['p95_ms_std']} | {row['error_rate_mean']} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", nargs=2, metavar=("SOURCE", "SUMMARY_CSV"), required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    all_summary: List[Dict[str, object]] = []
    all_objective: List[Dict[str, object]] = []
    for source, path_str in args.input:
        rows = read_csv(Path(path_str))
        all_summary.extend(summarize_source(source, rows))
        all_objective.extend(objective_table(source, rows))

    write_csv(out_dir / "k8s_replay_mean_std_summary.csv", all_summary)
    write_csv(out_dir / "k8s_replay_objective_disagreement.csv", all_objective)
    write_markdown(out_dir / "K8S_REPLAY_RECORD_AND_REPLAY_SUMMARY.md", all_summary, all_objective)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
