from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = {
    "last_value": 0,
    "dlinear": 1,
    "race_dlinear": 2,
    "patchtst": 3,
    "official_patchtst": 4,
    "official_itransformer": 5,
}


CASE_STUDIES = [
    {
        "id": "netman_missing_variables_30",
        "source": "netman",
        "dataset": "netman",
        "stress": "missing_variables_30",
        "title": "Telemetry outage: missing variables on NetMan KPI",
        "why": "Missing-variable stress simulates an entire monitoring metric becoming unavailable. This exposes telemetry-outage behavior that clean forecasting cannot measure.",
        "applied": "CloudOps model selection should treat metric outage as a first-class deployment condition, otherwise a clean-accurate model can be over-trusted during incidents.",
    },
    {
        "id": "gaia_changepoint_level_shift",
        "source": "gaia",
        "dataset": "changepoint",
        "stress": "level_shift",
        "title": "Concept shift: GAIA changepoint under level shift",
        "why": "Level-shift stress mimics releases, migrations, or workload-regime changes that alter the recent context before the forecast horizon.",
        "applied": "A model chosen only by clean MSE may not be the stable deployment choice when operational regimes shift.",
    },
    {
        "id": "netman_missing_30",
        "source": "netman",
        "dataset": "netman",
        "stress": "missing_30",
        "title": "Accuracy-latency tradeoff: NetMan 30% missing points",
        "why": "Point-wise missing telemetry reveals an accuracy-latency tradeoff: patch-based models can be more accurate but slower than linear deployment baselines.",
        "applied": "For a low-latency monitoring path, the minimum-MSE model is not automatically the best operational model.",
    },
    {
        "id": "alibaba2018_missing_variables_30",
        "source": "alibaba2018",
        "dataset": "alibaba2018",
        "stress": "missing_variables_30",
        "title": "Multivariate resource telemetry: Alibaba 2018 missing variables",
        "why": "Alibaba 2018 provides machine-level CPU, memory, network, and disk metrics. Under missing-variable stress, PatchTST-lite is more accurate but slower, while DLinear is fast but less robust.",
        "applied": "Resource-telemetry forecasting should compare accuracy, latency, memory, and outage robustness together, not clean accuracy alone.",
    },
]


def format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value == 0:
            return "0"
        if abs(value) >= 1_000_000 or abs(value) < 0.001:
            return f"{value:.3e}"
        return f"{value:.4g}"
    return str(value)


def to_markdown(frame: pd.DataFrame) -> str:
    headers = frame.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_value(row[col]) for col in headers) + " |")
    return "\n".join(lines)


def normalize_stress(frame: pd.DataFrame) -> pd.DataFrame:
    if "scenario" in frame.columns and "stress" not in frame.columns:
        frame = frame.rename(columns={"scenario": "stress"})
    frame["stress"] = frame["stress"].replace(
        {
            "missing_points_30": "missing_30",
            "noise_20": "noisy",
        }
    )
    return frame


def load_native_metrics(outputs: Path) -> pd.DataFrame:
    frames = []
    for source, path in [
        ("gaia", outputs / "gaia_matrix_core_summary.csv"),
        ("netman", outputs / "netman_kpi_core_summary.csv"),
        ("alibaba2018", outputs / "alibaba2018_machine_usage_core_summary.csv"),
    ]:
        if not path.exists():
            continue
        frame = normalize_stress(pd.read_csv(path))
        if "source" not in frame.columns:
            frame.insert(0, "source", source)
        if "dataset" not in frame.columns:
            frame.insert(1, "dataset", source)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_official_itransformer(official_dir: Path) -> pd.DataFrame:
    frames = []
    clean_path = official_dir / "itransformer_clean.csv"
    stress_path = official_dir / "itransformer_stress.csv"
    if clean_path.exists():
        clean = pd.read_csv(clean_path)
        clean["stress"] = "clean"
        frames.append(clean)
    if stress_path.exists():
        frames.append(pd.read_csv(stress_path))
    if not frames:
        return pd.DataFrame()
    frame = normalize_stress(pd.concat(frames, ignore_index=True))
    frame = frame.rename(columns={"dataset": "source"})
    frame["dataset"] = frame["source"]
    frame["model"] = "official_itransformer"
    return frame


def load_official_patchtst(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = normalize_stress(pd.read_csv(path))
    frame["model"] = "official_patchtst"
    if "source" not in frame.columns:
        frame["source"] = frame.get("dataset", "unknown")
    if "dataset" not in frame.columns:
        frame["dataset"] = frame["source"]
    return frame


def load_all_metrics(outputs: Path, official_dir: Path, official_patchtst: Path) -> pd.DataFrame:
    frames = [
        load_native_metrics(outputs),
        load_official_itransformer(official_dir),
        load_official_patchtst(official_patchtst),
    ]
    frames = [frame for frame in frames if not frame.empty]
    metrics = pd.concat(frames, ignore_index=True)
    for col in ["capacity_cost", "params", "max_memory_mb", "latency_p95_ms", "mse", "mae"]:
        if col not in metrics.columns:
            metrics[col] = np.nan
    metrics["model_order"] = metrics["model"].map(MODEL_ORDER).fillna(99)
    return metrics


def overlay_figures(outputs: Path, case_id: str) -> str:
    root = outputs / "paper_figures"
    return "; ".join(str(path) for path in sorted(root.glob(f"{case_id}_overlay.*")))


def write_case_sections(metrics: pd.DataFrame, outputs: Path, output_dir: Path) -> list[str]:
    sections: list[str] = []
    case_rows = []
    keep = [
        "source",
        "dataset",
        "stress",
        "model",
        "mse",
        "mae",
        "latency_p95_ms",
        "capacity_cost",
        "params",
        "max_memory_mb",
    ]
    for case in CASE_STUDIES:
        subset = metrics[
            (metrics["source"] == case["source"])
            & (metrics["dataset"] == case["dataset"])
            & (metrics["stress"] == case["stress"])
        ].copy()
        subset = subset.sort_values(["mse", "latency_p95_ms", "model_order"])
        case_path = output_dir / f"{case['id']}_metrics.csv"
        subset[keep].to_csv(case_path, index=False)
        figures = overlay_figures(outputs, case["id"])
        case_rows.append(
            {
                "id": case["id"],
                "title": case["title"],
                "metrics": str(case_path),
                "figures": figures,
                "why": case["why"],
                "applied": case["applied"],
            }
        )
        sections += [
            f"## {case['title']}",
            "",
            case["why"],
            "",
            to_markdown(subset[keep]),
            "",
            f"Figures: {figures}",
            "",
            f"Applied conclusion: {case['applied']}",
            "",
        ]

    pd.DataFrame(case_rows).to_csv(output_dir / "case_studies.csv", index=False)
    return sections


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a go/no-go decision report.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--official-dir", default="outputs/official_baselines")
    parser.add_argument("--official-patchtst", default="outputs/official_patchtst_native_summary.csv")
    parser.add_argument("--output-dir", default="outputs/decision_report")
    args = parser.parse_args()

    outputs = Path(args.outputs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_all_metrics(outputs, Path(args.official_dir), Path(args.official_patchtst))

    sections = [
        "# AIOps Forecasting Under Deployment Stress Decision Report",
        "",
        "Decision: **Go**, but frame the work as an Applied benchmark/system evaluation, not a new-model SOTA paper.",
        "",
    ]

    official_i = metrics[metrics["model"] == "official_itransformer"].copy()
    if not official_i.empty:
        clean = official_i[official_i["stress"] == "clean"]
        stress = official_i[official_i["stress"] != "clean"]
        clean.to_csv(output_dir / "official_itransformer_clean.csv", index=False)
        stress.to_csv(output_dir / "official_itransformer_stress.csv", index=False)
        sections += ["## Official iTransformer", "", to_markdown(official_i[["source", "stress", "mse", "mae", "latency_p95_ms", "params", "max_memory_mb"]]), ""]

    official_p = metrics[metrics["model"] == "official_patchtst"].copy()
    if not official_p.empty:
        official_p.to_csv(output_dir / "official_patchtst_native.csv", index=False)
        sections += [
            "## Official PatchTST Native Baseline",
            "",
            "Official PatchTST is imported from the official implementation and evaluated inside the native NPZ/stress pipeline. It is separate from the lightweight `patchtst` baseline.",
            "",
            to_markdown(official_p[["source", "stress", "mse", "mae", "latency_p95_ms", "capacity_cost", "params", "max_memory_mb"]]),
            "",
        ]

    sections += write_case_sections(metrics, outputs, output_dir)

    sections += [
        "## Reviewer Risks and Responses",
        "",
        "| risk | response |",
        "| --- | --- |",
        "| Looks like ordinary models on an application dataset. | The paper contribution is the deployment-stress protocol, public data audit, latency/memory reporting, capacity-risk proxy, and failure case studies, not a single model swap. |",
        "| Capacity proxy is too simple. | We add a forecast-to-capacity simulator with reactive baselines, under/over area, peak miss, and total normalized cost; it is still framed as a deployment proxy rather than a autoscaling system. |",
        "| GAIA and NetMan are not resource-capacity traces. | They are used for KPI stress diversity and telemetry-outage cases; Alibaba 2018 is the main multivariate resource-telemetry source. |",
        "| Alibaba is machine-level rather than service-level. | We state this limitation and avoid service-level request/SLO claims that the data cannot support. |",
        "| Official baseline comparability is imperfect. | Official PatchTST is evaluated inside the native NPZ/stress pipeline; official iTransformer is marked as an LTSF-bridge reference with protocol limitations. |",
        "| RACE-DLinear is not a strong new model. | We do not present it as SOTA; it is a lightweight robust baseline and ablation vehicle. |",
        "",
    ]

    gate = pd.DataFrame(
        [
            {"check": "official iTransformer clean/stress runs", "status": "done", "evidence": "outputs/official_baselines/itransformer_clean.csv; itransformer_stress.csv"},
            {"check": "official PatchTST native run", "status": "done", "evidence": "Alibaba clean, missing_10/30/50, missing_variables_30, delayed_12, noisy, burst, and level_shift are done."},
            {"check": "multivariate CloudOps source", "status": "done", "evidence": "Alibaba 2018 machine usage converted to data/alibaba2018_machine_usage.npz with shape [128,4096,5]."},
            {"check": "case studies", "status": "done", "evidence": "outputs/paper_figures/*_overlay.png and outputs/decision_report/case_studies.csv"},
            {"check": "forecast-to-capacity simulator", "status": "done", "evidence": "outputs/capacity_simulator_summary.csv; outputs/paper_tables/table_capacity_simulator.csv"},
            {"check": "benchmark manifest", "status": "done", "evidence": "benchmark_manifest.yaml"},
            {"check": "RACE as main method", "status": "not supported", "evidence": "RACE-DLinear is inconsistent; keep as lightweight baseline only."},
            {"check": "direction decision", "status": "go", "evidence": "Proceed as Applied benchmark/system evaluation, not as a new-model SOTA paper."},
        ]
    )
    gate.to_csv(output_dir / "decision_gate.csv", index=False)
    sections += ["## Decision Gate", "", to_markdown(gate), ""]

    (output_dir / "README.md").write_text("\n".join(sections), encoding="utf-8")
    print((output_dir / "README.md").resolve())
    print(to_markdown(gate))


if __name__ == "__main__":
    main()

