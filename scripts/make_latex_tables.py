from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_LABELS = {
    "last_observed": "LastObserved",
    "reactive_hpa": "Reactive HPA",
    "last_value": "LastValue",
    "dlinear": "DLinear",
    "race_dlinear": "RACE-DLinear",
    "patchtst": "PatchTST-lite",
    "official_patchtst": "Official PatchTST",
    "official_itransformer": "Official iTransformer",
    "official_itransformer_native": "Native iTransformer",
    "official_timemixer": "Official TimeMixer",
    "chronos_bolt_reference": "Chronos-Bolt ref.",
    "oracle": "Oracle",
}

SOURCE_LABELS = {
    "alibaba2018": "Alibaba",
    "salesforce_borg": "Salesforce/Borg",
}

STRESS_LABELS = {
    "clean": "Clean",
    "missing_30": "Missing points",
    "missing_variables_30": "Metric outage",
    "delayed_12": "Delayed tail",
    "level_shift": "Level shift",
    "burst": "Burst",
    "noisy": "Noise",
}


class RawTex(str):
    """String that should be inserted into generated LaTeX without escaping."""


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def fmt(value: object) -> str:
    if isinstance(value, RawTex):
        return str(value)
    if pd.isna(value):
        return "--"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if value == 0:
            return "0"
        if abs(value) >= 100000 or abs(value) < 0.001:
            return f"{value:.2e}"
        if abs(value) >= 100:
            return f"{value:.1f}"
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return tex_escape(value)


def fmt_pm(mean: object, std: object) -> str:
    return RawTex(f"{fmt(mean)} $\\pm$ {fmt(std)}")


def as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def model_label(value: object) -> str:
    return MODEL_LABELS.get(str(value), str(value))


def stress_label(value: object) -> str:
    return STRESS_LABELS.get(str(value), str(value))


def tabular(headers: list[str], rows: list[list[object]], align: str | None = None) -> str:
    if align is None:
        align = "l" + "r" * (len(headers) - 1)
    lines = [rf"\begin{{tabular}}{{{align}}}", r"\toprule"]
    lines.append(" & ".join(tex_escape(h) for h in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(fmt(value) for value in row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def table_env(label: str, caption: str, body: str, size: str = r"\scriptsize", resize: bool = False) -> str:
    wrapped_body = body
    if resize:
        wrapped_body = "\n".join([r"\resizebox{\columnwidth}{!}{%", body, r"}"])
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            size,
            rf"\caption{{{tex_escape(caption)}}}",
            rf"\label{{{label}}}",
            wrapped_body,
            r"\end{table}",
            "",
        ]
    )


def clean_model_name(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["model"] = frame["model"].map(MODEL_LABELS).fillna(frame["model"])
    return frame


def display_params(row: pd.Series, column: str = "params") -> object:
    if row.get("model") == "Chronos-Bolt ref.":
        return "pretrained"
    return row.get(column)


def display_memory(row: pd.Series, column: str = "memory_mb") -> object:
    if row.get("model") == "Official TimeMixer":
        value = row.get(column)
        if not pd.isna(value):
            return round(float(value))
    return row.get(column)


def make_dataset_table(table_dir: Path) -> str:
    frame = pd.read_csv(table_dir / "table_dataset_audit.csv")
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["dataset"],
                int(row["entities"]),
                int(row["time_steps"]),
                int(row["metrics"]),
                row["missing_rate"],
                row["zero_rate"],
            ]
        )
    body = tabular(["Dataset", "Entities", "Steps", "Metrics", "Missing", "Zero"], rows)
    return table_env("tab:dataset-audit", "Dataset audit summary.", body)


def make_benchmark_card_table(table_dir: Path) -> str:
    path = table_dir / "table_benchmark_card.csv"
    if not path.exists():
        return "\\label{tab:benchmark-card}% table_benchmark_card.csv not available yet\n"
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        rows.append([row["component"], row["setting"], row["details"]])
    body = tabular(["Component", "Setting", "Details"], rows, align="p{0.18\\linewidth}p{0.32\\linewidth}p{0.39\\linewidth}")
    return table_env("tab:benchmark-card", "AIOpsStressBench benchmark card.", body, size=r"\tiny", resize=True)


def make_baseline_fairness_table(table_dir: Path) -> str:
    path = table_dir / "table_baseline_fairness.csv"
    if not path.exists():
        return "\\label{tab:baseline-fairness}% table_baseline_fairness.csv not available yet\n"
    frame = pd.read_csv(path)
    rows = []
    selected = frame[frame["model"].isin(["LastValue", "DLinear", "RACE-DLinear", "PatchTST-lite", "Official PatchTST", "Native iTransformer", "Official TimeMixer", "Chronos-Bolt reference", "iTransformer bridge"])]
    for _, row in selected.iterrows():
        rows.append([row["model"], row["implementation"], row["protocol"], row["coverage"], row["role"]])
    body = tabular(["Model", "Implementation", "Protocol", "Coverage", "Role"], rows, align="p{0.16\\linewidth}p{0.21\\linewidth}p{0.25\\linewidth}p{0.13\\linewidth}p{0.16\\linewidth}")
    return table_env("tab:baseline-fairness", "Baseline fairness and protocol comparability card.", body, size=r"\tiny", resize=True)


def make_stress_taxonomy_table(_: Path) -> str:
    rows = [
        ["Missing points", "Random input values unavailable", "Packet loss, scraper gaps", "Robustness curve"],
        ["Metric outage", "Entire input metric channel masked", "Exporter failure, telemetry outage", "Outage stress"],
        ["Delayed tail", "Most recent k input steps hidden", "Ingestion lag, delayed monitoring", "Delay stress"],
        ["Noise", "Gaussian perturbation on input", "Aggregation or sensor instability", "Noise stress"],
        ["Burst", "Sparse positive spikes in input", "Workload spike, flash crowd", "Burst stress"],
        ["Level shift", "Context level changes before horizon", "Release, migration, mitigation", "Shift stress"],
    ]
    body = tabular(["Stress", "Protocol", "Operational source", "Evaluation"], rows, align="p{0.16\\linewidth}p{0.28\\linewidth}p{0.29\\linewidth}p{0.18\\linewidth}")
    return table_env("tab:stress-taxonomy", "Deployment-stress taxonomy and operational motivation.", body, size=r"\tiny", resize=True)


def make_clean_table(table_dir: Path) -> str:
    frame = clean_model_name(pd.read_csv(table_dir / "table_clean_accuracy.csv"))
    frame = frame[frame["source"] == "alibaba2018"].copy()
    rows = []
    for _, row in frame.iterrows():
        rows.append([row["model"], row["mse"], row["mae"], row["latency_p95_ms"], display_params(row), display_memory(row, "max_memory_mb")])
    body = tabular(["Model", "MSE", "MAE", "P95 ms", "Params", "Mem MB"], rows)
    return table_env("tab:clean-alibaba", "Clean Alibaba forecasting accuracy and deployment cost.", body, resize=True)


def make_stress_table(table_dir: Path) -> str:
    frame = clean_model_name(pd.read_csv(table_dir / "table_stress_robustness.csv"))
    frame = frame[frame["source"] == "alibaba2018"].copy()
    rows = []
    for _, row in frame.iterrows():
        rows.append([row["model"], row["mean_mse"], row["mean_mae"], row["worst_mse"], row["mean_p95_ms"], display_memory(row)])
    body = tabular(["Model", "Mean MSE", "Mean MAE", "Worst MSE", "P95 ms", "Mem MB"], rows)
    return table_env("tab:stress-alibaba", "Alibaba deployment-stress robustness.", body, resize=True)


def make_risk_table(table_dir: Path) -> str:
    frame = clean_model_name(pd.read_csv(table_dir / "table_capacity_risk_proxy.csv"))
    frame = frame[frame["source"].isin(["alibaba2018", "netman"])].copy()
    frame["source"] = frame["source"].replace({"alibaba2018": "Alibaba", "netman": "NetMan"})
    rows = []
    for _, row in frame.iterrows():
        rows.append([row["source"], row["model"], row["mean_under_rate"], row["mean_over_rate"], row["mean_capacity_cost"], row["mean_p95_ms"]])
    body = tabular(["Source", "Model", "Under", "Over", "Cost", "P95 ms"], rows, align="llrrrr")
    return table_env("tab:capacity-risk", "Capacity-risk proxy on Alibaba and NetMan stress scenarios.", body, size=r"\tiny", resize=True)


def make_capacity_simulator_table(table_dir: Path) -> str:
    frame = clean_model_name(pd.read_csv(table_dir / "table_capacity_simulator.csv"))
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    frame = frame[
        (frame["source"].isin(source_labels))
        & frame["model"].isin(["Reactive HPA", "DLinear", "RACE-DLinear", "PatchTST-lite"])
    ].copy()
    frame["source"] = frame["source"].map(source_labels)
    rows = []
    for _, row in frame.iterrows():
        p95 = "controller ref." if row["model"] == "Reactive HPA" else row["latency_p95_ms"]
        rows.append(
            [
                row["source"],
                row["model"],
                row["mean_total_cost"],
                row["mean_under_area"],
                row["mean_over_area"],
                row["mean_peak_miss"],
                p95,
            ]
        )
    body = tabular(["Source", "Model", "Cost", "Under", "Over", "Peak miss", "P95 ms"], rows, align="llrrrrr")
    return table_env("tab:capacity-simulator", "Forecast-to-capacity decision proxy on multivariate telemetry sources.", body, size=r"\tiny", resize=True)


def make_deployment_cost_table(table_dir: Path) -> str:
    frame = clean_model_name(pd.read_csv(table_dir / "table_deployment_cost.csv"))
    frame = frame[frame["source"] == "alibaba2018"].copy()
    rows = []
    for _, row in frame.iterrows():
        rows.append([row["model"], row["clean_p95_ms"], row["mean_p95_ms"], display_params(row), display_memory(row)])
    body = tabular(["Model", "Clean P95", "Mean P95", "Params", "Mem MB"], rows)
    return table_env("tab:deployment-cost", "Alibaba deployment cost across clean and stress runs.", body, resize=True)


def make_cross_source_winners_table(table_dir: Path) -> str:
    path = table_dir / "table_scenario_winners.csv"
    if not path.exists():
        return "\\label{tab:cross-source-winners}% table_scenario_winners.csv not available yet\n"
    frame = pd.read_csv(path)
    selected_stresses = ["clean", "delayed_12", "missing_30", "missing_variables_30"]
    frame = frame[
        frame["source"].isin(SOURCE_LABELS)
        & frame["stress"].isin(selected_stresses)
    ].copy()
    if frame.empty:
        return "\\label{tab:cross-source-winners}% no selected cross-source winner rows yet\n"
    frame["source_order"] = frame["source"].map({"alibaba2018": 0, "salesforce_borg": 1})
    frame["stress_order"] = frame["stress"].map({stress: i for i, stress in enumerate(selected_stresses)})
    frame = frame.sort_values(["source_order", "stress_order"])
    rows = []
    for _, row in frame.iterrows():
        winners = {
            row["best_mse_model"],
            row["best_latency_model"],
            row["best_capacity_model"],
        }
        rows.append(
            [
                SOURCE_LABELS.get(row["source"], row["source"]),
                stress_label(row["stress"]),
                "all learned",
                model_label(row["best_mse_model"]),
                model_label(row["best_latency_model"]),
                model_label(row["best_capacity_model"]),
                "Y" if len(winners) > 1 else "N",
            ]
        )
    body = tabular(["Source", "Stress", "Pool", "Best MSE", "Best P95", "Best decision", "Flip"], rows, align="llllllc")
    return table_env(
        "tab:cross-source-winners",
        "Cross-source scenario winners in the single-run all-model pool. The table tests whether best-MSE, lowest-P95 learned, and best decision-cost models agree under the same source and stress setting. LastValue, Chronos-Bolt, and LTSF-bridge are excluded from winner selection; multi-seed stability is reported in Table XII.",
        body,
        size=r"\tiny",
        resize=True,
    )


def make_capacity_sensitivity_compact_table(table_dir: Path) -> str:
    ratio_path = table_dir / "table_capacity_cost_ratio_sensitivity.csv"
    horizon_path = table_dir / "table_capacity_horizon_sensitivity.csv"
    if not ratio_path.exists():
        return "\\label{tab:capacity-sensitivity-compact}% table_capacity_cost_ratio_sensitivity.csv not available yet\n"
    ratio_frame = pd.read_csv(ratio_path)
    rows = []
    for ratio in ["2:1", "5:1", "10:1"]:
        selected = ratio_frame[
            ratio_frame["source"].isin(SOURCE_LABELS)
            & (ratio_frame["stress"] == "missing_variables_30")
            & (ratio_frame["cost_ratio"] == ratio)
        ].copy()
        winners = {}
        costs = {}
        for _, row in selected.iterrows():
            winners[row["source"]] = model_label(row["best_total_cost_model"])
            costs[row["source"]] = float(row["best_total_cost"])
        rows.append(
            [
                f"Cost {ratio}",
                winners.get("alibaba2018", "--"),
                costs.get("alibaba2018", np.nan),
                winners.get("salesforce_borg", "--"),
                costs.get("salesforce_borg", np.nan),
                "same" if len(set(winners.values())) == 1 and len(winners) == 2 else "diff.",
            ]
        )
    if horizon_path.exists():
        horizon_frame = pd.read_csv(horizon_path)
        selected = horizon_frame[
            horizon_frame["source"].isin(SOURCE_LABELS)
            & (horizon_frame["stress"] == "missing_variables_30")
            & (horizon_frame["pred_len"].astype(int) == 96)
            & horizon_frame["model"].isin(["dlinear", "race_dlinear", "patchtst", "last_value"])
        ].copy()
        winners = {}
        costs = {}
        for source, group in selected.groupby("source"):
            best = group.sort_values("total_cost", kind="mergesort").iloc[0]
            winners[source] = model_label(best["model"])
            costs[source] = float(best["total_cost"])
        rows.append(
            [
                "Horizon 96",
                winners.get("alibaba2018", "--"),
                costs.get("alibaba2018", np.nan),
                winners.get("salesforce_borg", "--"),
                costs.get("salesforce_borg", np.nan),
                "same" if len(set(winners.values())) == 1 and len(winners) == 2 else "diff.",
            ]
        )
    body = tabular(["Sensitivity", "Alibaba winner", "Alibaba cost", "Salesforce/Borg winner", "Salesforce/Borg cost", "Pattern"], rows, align="llrlrl")
    return table_env(
        "tab:capacity-sensitivity-compact",
        "Capacity-proxy sensitivity for metric-channel outage. Winners minimize total normalized decision cost within each source and setting.",
        body,
        size=r"\scriptsize",
        resize=True,
    )


def make_public_fault_mini_table(table_dir: Path) -> str:
    path = table_dir.parent / "public_fault_slice_v2" / "fault_slice_v2_compact.csv"
    if not path.exists():
        return "\\label{tab:public-fault-mini}% fault_slice_v2_compact.csv not available yet\n"
    frame = pd.read_csv(path)
    row = frame[frame["fault_type"] == "all"]
    if row.empty:
        return "\\label{tab:public-fault-mini}% no aggregate RE2-OB row yet\n"
    row = row.iloc[0]
    rows = [
        [
            "RE2-OB",
            "fault injection",
            int(row["cases"]),
            "Istio P99",
            int(row["ranking_changed"]),
            int(row["mse_decision_cost_disagreement"]),
            "sanity check",
        ]
    ]
    body = tabular(["Dataset", "Slice", "Cases", "Target", "MSE changes", "MSE/decision", "Role"], rows, align="llrllrl")
    return table_env(
        "tab:public-fault-mini",
        "External RE2-OB fault-window sanity check. It is not used for resource-capacity claims.",
        body,
        size=r"\scriptsize",
        resize=True,
    )


def make_stress_realism_proxy_table(table_dir: Path) -> str:
    path = table_dir / "table_stress_realism_audit.csv"
    if not path.exists():
        return "\\label{tab:stress-realism-proxy}% table_stress_realism_audit.csv not available yet\n"
    frame = pd.read_csv(path).set_index("dataset")
    rows = []
    proxy_rows = [
        ("Zero-run ratio", "long_zero_run12_channel_rate", "Metric outage"),
        ("Flatline windows", "flatline12_channel_rate", "Stale telemetry"),
        ("Robust spike rate", "spike_fraction_z6", "Burst/noise"),
        ("P95 level-shift score", "p95_level_shift_score", "Level shift"),
    ]
    for label, column, operator in proxy_rows:
        rows.append(
            [
                label,
                fmt(frame.loc["alibaba2018", column]),
                fmt(frame.loc["salesforce_borg", column]),
                operator,
            ]
        )
    body = tabular(["Public-trace proxy", "Alibaba", "Salesforce/Borg", "Stress operator motivated"], rows, align="lrrl")
    return table_env(
        "tab:stress-realism-proxy",
        "Compact stress-calibration proxies from public telemetry. Values motivate controlled operators but do not validate incident frequency.",
        body,
        size=r"\scriptsize",
        resize=True,
    )


def make_capacity_simulator_winner_table(table_dir: Path) -> str:
    frame = pd.read_csv(table_dir / "table_capacity_simulator_winners.csv")
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    frame = frame[
        (frame["source"].isin(source_labels))
        & frame["stress"].isin(["missing_30", "missing_variables_30", "delayed_12"])
    ].copy()
    frame["source"] = frame["source"].map(source_labels)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["source"],
                row["stress"],
                MODEL_LABELS.get(row["best_total_cost_model"], row["best_total_cost_model"]),
                MODEL_LABELS.get(row["best_under_area_model"], row["best_under_area_model"]),
                MODEL_LABELS.get(row["best_peak_miss_model"], row["best_peak_miss_model"]),
            ]
        )
    body = tabular(["Source", "Stress", "Best cost", "Best under", "Best peak"], rows, align="lllll")
    return table_env("tab:capacity-simulator-winners", "Forecast-to-capacity proxy winners by deployment objective.", body, size=r"\tiny", resize=True)


def make_capacity_horizon_table(table_dir: Path) -> str:
    path = table_dir / "table_capacity_horizon_sensitivity.csv"
    if not path.exists():
        return "\\label{tab:capacity-horizon}% table_capacity_horizon_sensitivity.csv not available yet\n"
    frame = pd.read_csv(path)
    frame = frame[
        (frame["source"] == "alibaba2018")
        & frame["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12"])
        & frame["model"].isin(["dlinear", "race_dlinear", "patchtst"])
    ].copy()
    if frame.empty:
        return "\\label{tab:capacity-horizon}% no selected horizon rows yet\n"
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["stress"],
                MODEL_LABELS.get(row["model"], row["model"]),
                row["pred_len"],
                row["total_cost"],
                row["under_area"],
                row["peak_miss"],
                row["p95_ms"],
            ]
        )
    body = tabular(["Stress", "Model", "Horizon", "Cost", "Under", "Peak", "P95 ms"], rows, align="llrrrrr")
    return table_env("tab:capacity-horizon", "Forecast-to-capacity horizon sensitivity on Alibaba.", body, size=r"\tiny", resize=True)


def make_capacity_headroom_table(table_dir: Path) -> str:
    path = table_dir / "table_capacity_headroom_sensitivity.csv"
    if not path.exists():
        return "\\label{tab:capacity-headroom}% table_capacity_headroom_sensitivity.csv not available yet\n"
    frame = pd.read_csv(path)
    frame = frame[
        (frame["source"] == "alibaba2018")
        & frame["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12"])
        & frame["model"].isin(["dlinear", "race_dlinear", "patchtst"])
    ].copy()
    if frame.empty:
        return "\\label{tab:capacity-headroom}% no selected headroom rows yet\n"
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["stress"],
                MODEL_LABELS.get(row["model"], row["model"]),
                row["headroom"],
                row["total_cost"],
                row["under_area"],
                row["over_area"],
                row["peak_miss"],
            ]
        )
    body = tabular(["Stress", "Model", "Headroom", "Cost", "Under", "Over", "Peak"], rows, align="llrrrrr")
    return table_env("tab:capacity-headroom", "Forecast-to-capacity headroom sensitivity on Alibaba.", body, size=r"\tiny", resize=True)


def make_deployment_guidelines_table(table_dir: Path) -> str:
    path = table_dir / "table_deployment_guidelines.csv"
    if not path.exists():
        return "\\label{tab:deployment-guidelines}% table_deployment_guidelines.csv not available yet\n"
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["deployment_constraint"],
                row["recommended_model"],
                row["fallback_or_reference"],
                row["evidence"],
            ]
        )
    body = tabular(["Constraint", "Primary", "Fallback", "Evidence"], rows, align="p{0.23\\linewidth}p{0.17\\linewidth}p{0.25\\linewidth}p{0.23\\linewidth}")
    return table_env("tab:deployment-guidelines", "Deployment model-selection guidelines from AIOpsStressBench evidence.", body, size=r"\tiny", resize=True)


def make_severity_table(table_dir: Path) -> str:
    path = table_dir / "table_severity_auc.csv"
    if not path.exists():
        return "\\label{tab:severity-auc}% table_severity_auc.csv not available yet\n"
    frame = pd.read_csv(path)
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    frame = frame[
        (frame["source"].isin(source_labels))
        & frame["stress_family"].isin(["missing_points", "missing_variables", "delayed_tail", "noise"])
        & frame["model"].isin(["dlinear", "race_dlinear", "patchtst", "official_patchtst", "official_itransformer_native"])
    ].copy()
    if frame.empty:
        return "\\label{tab:severity-auc}% table_severity_auc.csv has no selected rows yet\n"
    frame["source"] = frame["source"].map(source_labels)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["source"],
                row["stress_family"],
                MODEL_LABELS.get(row["model"], row["model"]),
                row["relative_mse_auc"],
                row["degradation_slope"],
                row["worst_relative_mse"],
                row["mean_p95_ms"],
            ]
        )
    body = tabular(["Source", "Stress", "Model", "AUC", "Slope", "Worst rel.", "P95 ms"], rows, align="lllrrrr")
    return table_env("tab:severity-auc", "Stress-severity degradation across Alibaba and Salesforce/Borg.", body, size=r"\tiny", resize=True)


def make_multiseed_table(table_dir: Path) -> str:
    path = table_dir / "table_multiseed_stability.csv"
    if not path.exists():
        return "\\label{tab:multiseed}% table_multiseed_stability.csv not available yet\n"
    frame = pd.read_csv(path)
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    frame = frame[
        (frame["source"].isin(source_labels))
        & frame["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12"])
        & frame["model"].isin(["dlinear", "race_dlinear", "patchtst"])
    ].copy()
    if frame.empty:
        return "\\label{tab:multiseed}% table_multiseed_stability.csv has no selected rows yet\n"
    frame["source"] = frame["source"].map(source_labels)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["source"],
                row["stress"],
                MODEL_LABELS.get(row["model"], row["model"]),
                fmt_pm(row["mse_mean"], row["mse_std"]),
                fmt_pm(row["capacity_cost_mean"], row["capacity_cost_std"]),
                fmt_pm(row["p95_ms_mean"], row["p95_ms_std"]),
                row["seeds"],
            ]
        )
    body = tabular(["Source", "Stress", "Model", "MSE", "Cap. cost", "P95 ms", "Seeds"], rows, align="llllllr")
    return table_env("tab:multiseed", "Multi-seed stability on core deployment scenarios.", body, size=r"\tiny", resize=True)


def make_multiseed_compact_table(table_dir: Path) -> str:
    path = table_dir / "table_multiseed_compact.csv"
    if not path.exists():
        return "\\label{tab:multiseed-compact}% table_multiseed_compact.csv not available yet\n"
    frame = pd.read_csv(path)
    if {"best_mse_model", "best_capacity_model", "lowest_latency_model", "mse_vs_capacity_flip"}.issubset(frame.columns):
        source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
        frame = frame[frame["dataset"].isin(source_labels)].copy()
        frame["source_label"] = frame["dataset"].map(source_labels)
        rows = []
        for _, row in frame.iterrows():
            p_value = row.get("paired_t_p", "")
            p_value = "--" if pd.isna(p_value) or p_value == "" else f"{float(p_value):.4f}"
            rows.append(
                [
                    row["source_label"],
                    stress_label(row["stress"]),
                    int(row.get("seeds", 5)),
                    MODEL_LABELS.get(row["best_mse_model"], row["best_mse_model"]),
                    MODEL_LABELS.get(row["best_capacity_model"], row["best_capacity_model"]),
                    MODEL_LABELS.get(row["lowest_latency_model"], row["lowest_latency_model"]),
                    "Y" if as_bool(row["mse_vs_capacity_flip"]) else "N",
                    p_value,
                ]
            )
        body = tabular(["Source", "Stress", "Seeds", "Best MSE", "Best decision", "Low P95", "MSE != decision", "p-value"], rows, align="llrllllc")
        return table_env(
            "tab:multiseed-compact",
            "Five-seed winner stability over all 10 core source-stress settings in the lightweight pool. Rows compare the best-MSE, best decision-cost, and lowest-P95 learned models; p-values are paired tests over decision cost for MSE-vs-decision flips.",
            body,
            size=r"\tiny",
            resize=True,
        )
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    frame = frame[frame["source"].isin(source_labels)].copy()
    frame["source"] = frame["source"].map(source_labels)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["source"],
                row["stress"],
                MODEL_LABELS.get(row["model"], row["model"]),
                fmt_pm(row["mse_mean"], row["mse_std"]),
                fmt_pm(row["capacity_cost_mean"], row["capacity_cost_std"]),
                fmt_pm(row["p95_ms_mean"], row["p95_ms_std"]),
                row["seeds"],
            ]
        )
    body = tabular(["Source", "Stress", "Model", "MSE", "Cap. cost", "P95 ms", "Seeds"], rows, align="llllllr")
    return table_env(
        "tab:multiseed-compact",
        "Compact multi-seed stability check for clean and metric-outage scenarios.",
        body,
        size=r"\tiny",
        resize=True,
    )


def make_stress_realism_table(table_dir: Path) -> str:
    path = table_dir / "table_stress_realism_audit.csv"
    if not path.exists():
        return "\\label{tab:stress-realism}% table_stress_realism_audit.csv not available yet\n"
    frame = pd.read_csv(path)
    source_labels = {
        "gaia": "GAIA",
        "netman": "NetMan",
        "alibaba2018": "Alibaba",
        "salesforce_borg": "Salesforce/Borg",
    }
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                source_labels.get(row["dataset"], row["dataset"]),
                row["nonfinite_rate"],
                row["zero_rate"],
                row["long_zero_run12_channel_rate"],
                row["flatline12_channel_rate"],
                row["spike_fraction_z6"],
                row["p95_level_shift_score"],
            ]
        )
    body = tabular(["Source", "Nonfinite", "Zero", "Zero-run12", "Flatline12", "Spike z6", "P95 shift"], rows)
    return table_env(
        "tab:stress-realism",
        "Stress realism audit over public operational telemetry. Flatline and tail proxies do not claim true ingestion delay.",
        body,
        size=r"\tiny",
        resize=True,
    )


def make_natural_degradation_slices_table(table_dir: Path) -> str:
    path = table_dir / "table_natural_degradation_slices.csv"
    if not path.exists():
        return "\\label{tab:natural-degradation-slices}% table_natural_degradation_slices.csv not available yet\n"
    frame = pd.read_csv(path)
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    proxy_labels = {
        "high_flatline_or_zero_channel": "flatline/zero-run",
        "high_spike_score": "spike z6",
        "high_level_shift_score": "level shift top 5\\%",
        "tail_flatline_proxy": "tail flatline12",
    }
    proxy_order = {name: idx for idx, name in enumerate(proxy_labels)}
    frame["source_order"] = frame["dataset"].map({"alibaba2018": 0, "salesforce_borg": 1}).fillna(99)
    frame["proxy_order"] = frame["proxy_slice"].map(proxy_order).fillna(99)
    frame = frame.sort_values(["source_order", "proxy_order"])
    grouped = frame.groupby(["dataset", "source_order"], as_index=False).agg(
        proxy_slices=("proxy_slice", "count"),
        min_windows=("window_count", "min"),
        normal_to_slice_mse_flips=("normal_to_slice_mse_flip", "sum"),
        normal_to_slice_capacity_flips=("normal_to_slice_capacity_flip", "sum"),
        objective_flips=("mse_vs_capacity_flip", "sum"),
    )
    grouped = grouped.sort_values("source_order")
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            [
                source_labels.get(row["dataset"], row["dataset"]),
                int(row["proxy_slices"]),
                int(row["min_windows"]),
                int(row["normal_to_slice_mse_flips"]),
                int(row["normal_to_slice_capacity_flips"]),
                int(row["objective_flips"]),
            ]
        )
    body = tabular(
        ["Source", "Slices", "N/slice", "MSE flips", "Cap. flips", "Obj. flips"],
        rows,
        align="lrrrrr",
    )
    return table_env(
        "tab:natural-degradation-slices",
        "Natural degradation proxy slices without synthetic stress injection. Full per-slice winners are in the artifact CSV.",
        body,
        size=r"\tiny",
        resize=True,
    )


def make_imputation_pipeline_table(table_dir: Path) -> str:
    path = table_dir / "table_imputation_pipeline.csv"
    if not path.exists():
        return "\\label{tab:imputation-pipeline}% table_imputation_pipeline.csv not available yet\n"
    frame = pd.read_csv(path)
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    imputation_labels = {"none": "None", "ffill": "Forward fill", "mean": "Mean fill"}
    selected_rows = []
    ali = frame[
        (frame["source"] == "alibaba2018")
        & frame["stress"].isin(["missing_30", "missing_variables_30"])
        & (frame["model"] == "dlinear")
        & frame["imputation"].isin(["none", "ffill", "mean"])
    ].copy()
    selected_rows.append(ali)
    for stress in ["missing_30", "missing_variables_30"]:
        sf = frame[
            (frame["source"] == "salesforce_borg")
            & (frame["stress"] == stress)
            & (frame["model"] == "patchtst")
            & frame["imputation"].isin(["none", "ffill", "mean"])
        ].copy()
        none = sf[sf["imputation"] == "none"]
        fills = sf[sf["imputation"].isin(["ffill", "mean"])].sort_values("capacity_cost", kind="mergesort").head(1)
        selected_rows.append(pd.concat([none, fills], ignore_index=True))
    selected = pd.concat(selected_rows, ignore_index=True)
    selected["source_label"] = selected["source"].map(source_labels)
    selected["model_label"] = selected["model"].map(MODEL_LABELS).fillna(selected["model"])
    selected["imputation_label"] = selected["imputation"].map(imputation_labels).fillna(selected["imputation"])
    selected["source_order"] = selected["source"].map({"alibaba2018": 0, "salesforce_borg": 1}).fillna(99)
    selected["stress_order"] = selected["stress"].map({"missing_30": 0, "missing_variables_30": 1}).fillna(99)
    selected["imputation_order"] = selected["imputation"].map({"none": 0, "ffill": 1, "mean": 2}).fillna(99)
    selected = selected.sort_values(["source_order", "stress_order", "imputation_order"])
    rows = []
    for _, row in selected.iterrows():
        rows.append(
            [
                row["source_label"],
                row["stress"],
                row["model_label"],
                row["imputation_label"],
                row["mse"],
                row["capacity_cost"],
                row["latency_p95_ms"],
            ]
        )
    body = tabular(["Source", "Stress", "Model", "Preprocess", "MSE", "Cap. cost", "P95 ms"], rows, align="llllrrr")
    return table_env(
        "tab:imputation-pipeline",
        "Compact imputation check. Simple fill helps point missingness but does not repair whole metric-channel outage.",
        body,
        size=r"\tiny",
        resize=True,
    )


def make_stressroute_table(table_dir: Path) -> str:
    path = table_dir / "table_stressroute_v1.csv"
    if not path.exists():
        return "\\label{tab:stressroute}% table_stressroute_v1.csv not available yet\n"
    frame = pd.read_csv(path)
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    frame = frame[
        (frame["source"].isin(source_labels))
        & frame["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12"])
        & frame["latency_budget_ms"].isin([0.2, 1.0])
    ].copy()
    frame["source"] = frame["source"].map(source_labels)
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["source"],
                row["stress"],
                row["latency_budget_ms"],
                MODEL_LABELS.get(row["selected_model"], row["selected_model"]),
                row["capacity_cost"],
                row["capacity_cost_vs_dlinear"],
                row["latency_p95_ms"],
            ]
        )
    body = tabular(["Source", "Stress", "Budget", "Selected", "Cost", "Cost vs DLinear", "P95 ms"], rows, align="lllrrrr")
    return table_env("tab:stressroute", "StressRoute v1 deployment policy under latency budgets.", body, size=r"\tiny", resize=True)


def make_stressroute_v2_table(table_dir: Path) -> str:
    path = table_dir / "table_stressroute_v2.csv"
    if not path.exists():
        return "\\label{tab:stressroute-v2}% table_stressroute_v2.csv not available yet\n"
    frame = pd.read_csv(path)
    source_labels = {"alibaba2018": "Alibaba", "salesforce_borg": "Salesforce/Borg"}
    policy_labels = {
        "fixed_dlinear": "Fixed DLinear",
        "fixed_patchtst": "Fixed PatchTST-lite",
        "stressroute_v1": "StressRoute v1",
        "stressroute_v2": "StressRoute v2",
        "oracle": "Oracle",
    }
    frame = frame[
        (frame["source"].isin(source_labels))
        & (frame["stress"].isin(["missing_30", "missing_variables_30", "delayed_12", "burst", "level_shift"]))
        & (frame["latency_budget_ms"].isin([0.2, 0.5, 1.0]))
        & (frame["policy"].isin(policy_labels))
    ].copy()
    if frame.empty:
        return "\\label{tab:stressroute-v2}% no selected StressRoute v2 rows yet\n"
    frame["source_label"] = frame["source"].map(source_labels)
    frame["policy_label"] = frame["policy"].map(policy_labels)
    policy_order = {
        "fixed_dlinear": 0,
        "fixed_patchtst": 1,
        "stressroute_v1": 2,
        "stressroute_v2": 3,
        "oracle": 4,
    }

    def summarize_selection(values: pd.Series) -> str:
        cleaned = values.dropna().astype(str)
        if cleaned.empty:
            return "--"
        counts = cleaned.value_counts()
        if len(counts) == 1:
            return MODEL_LABELS.get(counts.index[0], counts.index[0])
        top = counts.index[0]
        return f"{MODEL_LABELS.get(top, top)}+"

    grouped = (
        frame.groupby(["source", "source_label", "latency_budget_ms", "policy", "policy_label"], as_index=False)
        .agg(
            selected=("selected_model", summarize_selection),
            capacity_cost=("capacity_cost", "mean"),
            capacity_cost_vs_dlinear=("capacity_cost_vs_dlinear", "mean"),
            latency_p95_ms=("latency_p95_ms", "mean"),
            budget_feasible=("budget_feasible", "min"),
            latency_vs_patchtst=("latency_vs_patchtst", "mean"),
            latency_constrained_regret=("latency_constrained_regret", "mean"),
            capacity_oracle_gap=("capacity_oracle_gap", "mean"),
            route_model_count=("route_model_count", "mean"),
        )
    )
    grouped["source_order"] = grouped["source"].map({"alibaba2018": 0, "salesforce_borg": 1}).fillna(99)
    grouped["policy_order"] = grouped["policy"].map(policy_order).fillna(99)
    grouped = grouped.sort_values(["source_order", "latency_budget_ms", "policy_order"])
    rows = []
    for _, row in grouped.iterrows():
        rows.append(
            [
                row["source_label"],
                row["latency_budget_ms"],
                row["policy_label"],
                row["selected"],
                "Y" if bool(row["budget_feasible"]) else "N",
                row["capacity_cost"],
                row["capacity_cost_vs_dlinear"],
                row["latency_p95_ms"],
                row["latency_vs_patchtst"],
                row["latency_constrained_regret"],
                row["capacity_oracle_gap"],
            ]
        )
    body = tabular(["Source", "Budget", "Policy", "Selected", "Feas.", "Cost", "vs DLinear", "P95", "Lat. vs PatchTST", "Regret", "Oracle gap"], rows, align="lllllrrrrrr")
    return table_env("tab:stressroute-v2", "StressRoute v2 policy comparison averaged over mixed deployment stresses.", body, size=r"\tiny", resize=True)


def make_stressroute_policy_table(table_dir: Path) -> str:
    path = table_dir / "table_stressroute_policy_summary.csv"
    if not path.exists():
        return "\\label{tab:stressroute-policy}% table_stressroute_policy_summary.csv not available yet\n"
    rows = []
    rows.append(["0.2 ms", "DLinear only", "DLinear", "Strict online alerting; latency dominates capacity objective."])
    rows.append(["0.5 ms", "DLinear / RACE-DLinear", "DLinear or RACE-DLinear", "Middle budget; keep lightweight models unless stress cost is high."])
    rows.append(["1.0 ms", "DLinear / RACE-DLinear / PatchTST-lite", "PatchTST-lite often feasible", "Relaxed budget; patch-based route can be justified under outage or delayed telemetry."])
    body = tabular(["Budget", "Feasible learned models", "Typical route", "Deployment interpretation"], rows, align="llll")
    return table_env(
        "tab:stressroute-policy",
        "StressRoute deployment-policy summary. Source-specific route counts, regret, and oracle-gap details are reported in the reproducibility package.",
        body,
        size=r"\scriptsize",
        resize=True,
    )


def make_structural_findings_table(table_dir: Path) -> str:
    path = table_dir / "table_structural_findings.csv"
    if not path.exists():
        return "\\label{tab:structural-findings}% table_structural_findings.csv not available yet\n"
    frame = pd.read_csv(path)
    rows = []
    for _, row in frame.iterrows():
        rows.append([row["finding"], row["evidence"], row["deployment_implication"]])
    body = tabular(["Finding", "Quantitative evidence", "Deployment implication"], rows, align="p{0.24\\linewidth}p{0.34\\linewidth}p{0.31\\linewidth}")
    return table_env(
        "tab:structural-findings",
        "Structural findings extracted from generated severity tables (table_severity_auc/table_severity_slope) and winner analyses.",
        body,
        size=r"\tiny",
        resize=True,
    )


def make_winner_table(table_dir: Path) -> str:
    frame = pd.read_csv(table_dir / "table_scenario_winners.csv")
    frame = frame[(frame["source"] == "alibaba2018") & frame["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12", "level_shift"])].copy()
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["stress"],
                MODEL_LABELS.get(row["best_mse_model"], row["best_mse_model"]),
                MODEL_LABELS.get(row["best_latency_model"], row["best_latency_model"]),
                MODEL_LABELS.get(row["best_capacity_model"], row["best_capacity_model"]),
            ]
        )
    body = tabular(["Stress", "Best MSE", "Best learned P95", "Best Capacity"], rows, align="llll")
    return table_env(
        "tab:scenario-winners",
        "Alibaba scenario winners within the core learned comparable pool: DLinear, RACE-DLinear, PatchTST-lite, official PatchTST, native iTransformer, and TimeMixer.",
        body,
        resize=True,
    )


def make_gaia_category_table(table_dir: Path) -> str:
    frame = pd.read_csv(table_dir / "table_gaia_category_analysis.csv")
    frame = frame[(frame["stress"] == "missing_30") & frame["dataset"].isin(["periodic", "changepoint", "low_snr", "partially_stationary"])].copy()
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            [
                row["dataset"],
                MODEL_LABELS.get(row["best_mse_model"], row["best_mse_model"]),
                MODEL_LABELS.get(row["best_latency_model"], row["best_latency_model"]),
                MODEL_LABELS.get(row["best_capacity_model"], row["best_capacity_model"]),
            ]
        )
    body = tabular(["GAIA subset", "Best MSE", "Best P95", "Best Cap."], rows, align="llll")
    return table_env("tab:gaia-category", "GAIA missing-30 winners by behavior category.", body, resize=True)


def make_ablation_table(table_dir: Path) -> str:
    frame = clean_model_name(pd.read_csv(table_dir / "table_race_ablation.csv"))
    rows = []
    for _, row in frame.iterrows():
        run = re.sub(r"^gaia_ablation_", "", str(row["run"])).replace("_", " ")
        rows.append([run, row["model"], row["mse"], row["mae"], row["latency_p95_ms"], row["capacity_cost"]])
    body = tabular(["Run", "Model", "MSE", "MAE", "P95 ms", "Cap. cost"], rows, align="llrrrr")
    return table_env("tab:ablation", "RACE-DLinear ablation on GAIA missing-30.", body, resize=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX table snippets for the manuscript.")
    parser.add_argument("--table-dir", default="outputs/paper_tables")
    parser.add_argument("--output-dir", default="paper/tables")
    args = parser.parse_args()

    table_dir = Path(args.table_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "benchmark_card.tex": make_benchmark_card_table(table_dir),
        "baseline_fairness.tex": make_baseline_fairness_table(table_dir),
        "dataset_audit.tex": make_dataset_table(table_dir),
        "stress_taxonomy.tex": make_stress_taxonomy_table(table_dir),
        "clean_alibaba.tex": make_clean_table(table_dir),
        "stress_alibaba.tex": make_stress_table(table_dir),
        "deployment_cost.tex": make_deployment_cost_table(table_dir),
        "cross_source_winners.tex": make_cross_source_winners_table(table_dir),
        "capacity_sensitivity_compact.tex": make_capacity_sensitivity_compact_table(table_dir),
        "public_fault_mini.tex": make_public_fault_mini_table(table_dir),
        "stress_realism_proxy.tex": make_stress_realism_proxy_table(table_dir),
        "capacity_risk.tex": make_risk_table(table_dir),
        "capacity_simulator.tex": make_capacity_simulator_table(table_dir),
        "capacity_simulator_winners.tex": make_capacity_simulator_winner_table(table_dir),
        "capacity_horizon.tex": make_capacity_horizon_table(table_dir),
        "capacity_headroom.tex": make_capacity_headroom_table(table_dir),
        "deployment_guidelines.tex": make_deployment_guidelines_table(table_dir),
        "severity_auc.tex": make_severity_table(table_dir),
        "multiseed.tex": make_multiseed_table(table_dir),
        "multiseed_compact.tex": make_multiseed_compact_table(table_dir),
        "stress_realism_audit.tex": make_stress_realism_table(table_dir),
        "natural_degradation_slices.tex": make_natural_degradation_slices_table(table_dir),
        "imputation_pipeline.tex": make_imputation_pipeline_table(table_dir),
        "stressroute.tex": make_stressroute_table(table_dir),
        "stressroute_v2.tex": make_stressroute_v2_table(table_dir),
        "stressroute_policy.tex": make_stressroute_policy_table(table_dir),
        "structural_findings.tex": make_structural_findings_table(table_dir),
        "scenario_winners.tex": make_winner_table(table_dir),
        "gaia_category.tex": make_gaia_category_table(table_dir),
        "ablation.tex": make_ablation_table(table_dir),
    }
    for name, content in tables.items():
        path = output_dir / name
        path.write_text(content, encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
