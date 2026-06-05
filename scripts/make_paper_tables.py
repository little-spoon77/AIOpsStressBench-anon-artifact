from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_ORDER = {
    "last_observed": -2,
    "reactive_hpa": -1,
    "last_value": 0,
    "dlinear": 1,
    "race_dlinear": 2,
    "race_dlinear_nomask": 3,
    "patchtst": 4,
    "official_patchtst": 5,
    "official_itransformer": 6,
    "official_itransformer_native": 7,
    "official_timemixer": 8,
    "chronos_bolt_reference": 9,
}

CORE_LEARNED_MODELS = {
    "dlinear",
    "race_dlinear",
    "patchtst",
    "official_patchtst",
    "official_itransformer_native",
    "official_timemixer",
}


STRESS_ALIASES = {
    "missing_points_30": "missing_30",
    "noise_20": "noisy",
}


def format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        abs_value = abs(value)
        if abs_value == 0:
            return "0"
        if abs_value >= 1_000_000 or abs_value < 0.001:
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


def write_table(frame: pd.DataFrame, output_dir: Path, stem: str) -> None:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(frame) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


def load_metrics(path: Path, source: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "scenario" in frame.columns and "stress" not in frame.columns:
        frame = frame.rename(columns={"scenario": "stress"})
    if "dataset" not in frame.columns:
        frame.insert(0, "dataset", source)
    frame.insert(0, "source", source)
    frame["model_order"] = frame["model"].map(MODEL_ORDER).fillna(99)
    return frame


def load_official_metrics(output_dir: Path) -> pd.DataFrame:
    frames = []
    clean_path = output_dir / "itransformer_clean.csv"
    stress_path = output_dir / "itransformer_stress.csv"
    if clean_path.exists():
        clean = pd.read_csv(clean_path)
        clean["stress"] = "clean"
        frames.append(clean)
    if stress_path.exists():
        stress = pd.read_csv(stress_path)
        stress["stress"] = stress["stress"].replace(STRESS_ALIASES)
        frames.append(stress)
    if not frames:
        return pd.DataFrame()

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.rename(columns={"dataset": "source"})
    frame["dataset"] = frame["source"]
    frame["model"] = "official_itransformer"
    frame["model_order"] = frame["model"].map(MODEL_ORDER).fillna(99)
    for col in [
        "capacity_under_rate",
        "capacity_over_rate",
        "capacity_mean_under",
        "capacity_mean_over",
        "capacity_cost",
    ]:
        if col not in frame.columns:
            frame[col] = np.nan
    return frame


def load_official_patchtst(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["model"] = "official_patchtst"
    if "source" not in frame.columns:
        frame["source"] = frame.get("dataset", "unknown")
    if "dataset" not in frame.columns:
        frame["dataset"] = frame["source"]
    frame["model_order"] = frame["model"].map(MODEL_ORDER).fillna(99)
    return frame


def build_deployment_guidelines() -> pd.DataFrame:
    rows = [
        {
            "deployment_constraint": "Strict online latency",
            "recommended_model": "DLinear",
            "fallback_or_reference": "LastValue as a zero-parameter lower bound",
            "evidence": "Cost; winners",
            "rationale": "Among learned forecasters, DLinear has the lowest P95 latency and parameter count, so it is the safest default for online alerting paths.",
        },
        {
            "deployment_constraint": "Telemetry outage / missing variables",
            "recommended_model": "PatchTST-lite",
            "fallback_or_reference": "DLinear when latency is the hard constraint",
            "evidence": "Stress; stability; cases",
            "rationale": "Patch-based context modeling is more robust on Alibaba missing-variable stress, but it costs higher latency than DLinear.",
        },
        {
            "deployment_constraint": "Low under-provision tolerance",
            "recommended_model": "DLinear with higher headroom",
            "fallback_or_reference": "PatchTST-lite for missing-variable windows",
            "evidence": "Capacity winners; headroom",
            "rationale": "Capacity objectives often select DLinear for total cost and under-provision area, while missing-variable windows can favor PatchTST-lite.",
        },
        {
            "deployment_constraint": "Offline capacity planning",
            "recommended_model": "DLinear",
            "fallback_or_reference": "PatchTST-lite if telemetry quality is poor",
            "evidence": "Proxy; horizon",
            "rationale": "Forecast-to-capacity proxy favors low-cost DLinear in many stress settings, but the best policy changes under outage and delayed telemetry.",
        },
        {
            "deployment_constraint": "Long horizon or delayed telemetry",
            "recommended_model": "Re-evaluate by horizon",
            "fallback_or_reference": "DLinear for total cost; PatchTST-lite for selected peak-miss cases",
            "evidence": "Horizon; capacity winners",
            "rationale": "Longer horizons increase peak-miss and total cost, so the selected model should be chosen by deployment objective rather than clean MSE.",
        },
        {
            "deployment_constraint": "Memory-constrained edge path",
            "recommended_model": "DLinear",
            "fallback_or_reference": "RACE-DLinear only as a robust lightweight baseline",
            "evidence": "Cost; ablation",
            "rationale": "Linear baselines keep parameters and memory low; RACE-DLinear should not be claimed as a universal leading model.",
        },
    ]
    return pd.DataFrame(rows)


def build_benchmark_card() -> pd.DataFrame:
    rows = [
        {
            "component": "Task",
            "setting": "Operational forecasting under deployment stress",
            "details": "Input length 96, horizon 24, chronological 65/15/20 split.",
        },
        {
            "component": "Public sources",
            "setting": "GAIA, NetMan, Alibaba 2018, Salesforce/Borg 2011",
            "details": "Alibaba and Salesforce/Borg are multivariate telemetry; GAIA/NetMan provide KPI stress diversity.",
        },
        {
            "component": "Stress operators",
            "setting": "Missing points, metric outage, delayed tail, noise, burst, level shift",
            "details": "Stress is injected only into the input window; targets remain clean.",
        },
        {
            "component": "Metrics",
            "setting": "MSE/MAE plus deployment metrics",
            "details": "P95 latency, parameters, memory, severity slope, and forecast-to-capacity proxy.",
        },
        {
            "component": "Core comparable pool",
            "setting": "LastValue, DLinear, RACE-DLinear, PatchTST-lite, official PatchTST, native iTransformer, TimeMixer",
            "details": "Native or native-wrapper stress protocol where available; RACE-DLinear is a lightweight robust baseline, not a claimed leading model.",
        },
        {
            "component": "Reference-only pool",
            "setting": "Chronos-Bolt zero-shot, LTSF-bridge iTransformer",
            "details": "Feasibility and context only; these rows do not drive the central model-ranking claims.",
        },
        {
            "component": "Reproducibility",
            "setting": "Minimal and extended reproduction paths",
            "details": "Minimal path regenerates the core winner and latency tables; extended path regenerates severity, sensitivity, and multi-seed results.",
        },
    ]
    return pd.DataFrame(rows)


def build_baseline_fairness(frame: pd.DataFrame) -> pd.DataFrame:
    coverage = {}
    for model, group in frame.groupby("model", sort=False):
        coverage[model] = f"{group['source'].nunique()} src / {group['stress'].nunique()} scenarios"
    rows = [
        {
            "model": "LastValue",
            "implementation": "Native deterministic baseline",
            "protocol": "Same NPZ windows and stress operators",
            "coverage": coverage.get("last_value", "not run"),
            "budget": "No training; zero parameters",
            "role": "Latency lower bound, not a learned forecaster",
        },
        {
            "model": "DLinear",
            "implementation": "Native supervised baseline",
            "protocol": "Same NPZ windows, stress injection, and capacity evaluator",
            "coverage": coverage.get("dlinear", "not run"),
            "budget": "Shared epochs, batch size, seed, and hardware",
            "role": "Low-cost learned deployment baseline",
        },
        {
            "model": "RACE-DLinear",
            "implementation": "Native lightweight robust baseline",
            "protocol": "Same NPZ windows and stress-aware training",
            "coverage": coverage.get("race_dlinear", "not run"),
            "budget": "Shared budget with DLinear",
            "role": "Ablation vehicle; not a leading-model claim",
        },
        {
            "model": "PatchTST-lite",
            "implementation": "Native lightweight patch backbone",
            "protocol": "Same NPZ windows, stress injection, and latency profiler",
            "coverage": coverage.get("patchtst", "not run"),
            "budget": "Shared native training budget",
            "role": "Patch-family robustness reference",
        },
        {
            "model": "Official PatchTST",
            "implementation": "Official model class in native wrapper",
            "protocol": "Native NPZ/stress pipeline when available",
            "coverage": coverage.get("official_patchtst", "not run"),
            "budget": "Limited official run budget; reported as reference",
            "role": "Official patch-family comparability check",
        },
        {
            "model": "Native iTransformer",
            "implementation": "Official model class in native wrapper",
            "protocol": "Native NPZ/stress wrapper for selected core scenarios",
            "coverage": coverage.get("official_itransformer_native", "not run"),
            "budget": "Limited official run budget; reported as reference",
            "role": "Transformer reference under comparable wrapper",
        },
        {
            "model": "Official TimeMixer",
            "implementation": "THUML Time-Series-Library model in native wrapper",
            "protocol": "Native NPZ/stress wrapper for core scenarios",
            "coverage": coverage.get("official_timemixer", "not run"),
            "budget": "Limited official run budget; reported as reference",
            "role": "Multiscale-mixing reference for model-pool coverage",
        },
        {
            "model": "Chronos-Bolt reference",
            "implementation": "chronos-forecasting zero-shot pipeline",
            "protocol": "Target-metric zero-shot subset; no fine-tuning",
            "coverage": coverage.get("chronos_bolt_reference", "not run"),
            "budget": "512-window feasibility reference",
            "role": "Foundation-model latency and robustness context",
        },
        {
            "model": "iTransformer bridge",
            "implementation": "Official LTSF CSV bridge",
            "protocol": "Flattened entity-metric channels; reference only",
            "coverage": coverage.get("official_itransformer", "not run"),
            "budget": "Separate LTSF-style setup",
            "role": "Not used for core native ranking",
        },
    ]
    return pd.DataFrame(rows)


def main_comparison_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep core protocol-comparable rows for main paper rankings.

    The LTSF-bridge iTransformer rows are useful reference analysis, but they
    flatten entity-metric channels and should not drive the main ranking tables.
    """
    return frame[frame["model"].ne("official_itransformer")].copy()


def core_learned_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows eligible for main learned-forecaster winner claims."""
    return frame[frame["model"].isin(CORE_LEARNED_MODELS)].copy()


def load_native_official(path: Path, model_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["model"] = model_name
    if "source" not in frame.columns:
        frame["source"] = frame.get("dataset", "unknown")
    if "dataset" not in frame.columns:
        frame["dataset"] = frame["source"]
    frame["model_order"] = frame["model"].map(MODEL_ORDER).fillna(99)
    return frame


def load_official_timemixer(path: Path) -> pd.DataFrame:
    return load_native_official(path, "official_timemixer")


def load_capacity_simulator(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "pred_len" not in frame.columns:
        frame["pred_len"] = 24
    if "headroom" not in frame.columns:
        frame["headroom"] = 0.15
    frame["model_order"] = frame["model"].map(MODEL_ORDER).fillna(99)
    return frame


def load_severity(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["model_order"] = frame["model"].map(MODEL_ORDER).fillna(99)
    return frame


def load_multiseed(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["model_order"] = frame["model"].map(MODEL_ORDER).fillna(99)
    return frame


def load_stressroute(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame["selected_model_order"] = frame["selected_model"].map(MODEL_ORDER).fillna(99)
    return frame


def audit_npz(name: str, path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    series = data["series"].astype(np.float32)
    finite = np.isfinite(series)
    return {
        "dataset": name,
        "entities": series.shape[0],
        "time_steps": series.shape[1],
        "metrics": series.shape[2],
        "finite_rate": float(finite.mean()),
        "missing_rate": float(1.0 - finite.mean()),
        "zero_rate": float(np.mean(series[finite] == 0.0)) if finite.any() else np.nan,
        "mean": float(np.nanmean(series)),
        "std": float(np.nanstd(series)),
        "min": float(np.nanmin(series)),
        "max": float(np.nanmax(series)),
        "path": str(path),
    }


def add_best_columns(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        for metric, label in [
            ("mse", "best_mse_model"),
            ("mae", "best_mae_model"),
            ("latency_p95_ms", "best_latency_model"),
            ("capacity_cost", "best_capacity_model"),
        ]:
            if metric in group:
                winner = group.sort_values([metric, "model_order"]).iloc[0]
                row[label] = winner["model"]
                row[f"{metric}_best"] = winner[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_win_rates(frame: pd.DataFrame) -> pd.DataFrame:
    win_rows = []
    for metric in ["mse", "mae", "latency_p95_ms", "capacity_cost"]:
        usable = frame.dropna(subset=[metric])
        if usable.empty:
            continue
        winners = (
            usable.sort_values([metric, "model_order"])
            .groupby(["source", "dataset", "stress"], as_index=False)
            .first()
        )
        counts = winners.groupby(["source", "model"]).size().reset_index(name=f"{metric}_wins")
        win_rows.append(counts)
    if not win_rows:
        return pd.DataFrame(columns=["source", "model"])
    result = win_rows[0]
    for extra in win_rows[1:]:
        result = result.merge(extra, on=["source", "model"], how="outer")
    count_cols = [c for c in result.columns if c.endswith("_wins")]
    result[count_cols] = result[count_cols].fillna(0).astype(int)
    result["model_order"] = result["model"].map(MODEL_ORDER).fillna(99)
    return result.sort_values(["source", "model_order"]).drop(columns=["model_order"])


def summarize_severity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows = []
    slope_rows = []
    for keys, group in frame.groupby(["source", "dataset", "stress_family", "model"], sort=True):
        source, dataset, family, model = keys
        group = group.sort_values("level").copy()
        clean = group[group["level"].astype(float) == 0.0]
        if clean.empty:
            continue
        clean_mse = float(clean.iloc[0]["mse"])
        group["relative_mse"] = group["mse"] / clean_mse if clean_mse != 0 else np.nan
        integrate = getattr(np, "trapezoid", np.trapz)
        auc = float(integrate(group["relative_mse"], group["level"])) if len(group) > 1 else 0.0
        if len(group) > 1:
            slope = float(np.polyfit(group["level"].astype(float), group["relative_mse"].astype(float), deg=1)[0])
        else:
            slope = 0.0
        worst = group.iloc[group["relative_mse"].astype(float).argmax()]
        row = {
            "source": source,
            "dataset": dataset,
            "stress_family": family,
            "model": model,
            "clean_mse": clean_mse,
            "relative_mse_auc": auc,
            "degradation_slope": slope,
            "worst_level": worst["level"],
            "worst_relative_mse": worst["relative_mse"],
            "mean_p95_ms": group["latency_p95_ms"].mean(),
            "model_order": MODEL_ORDER.get(model, 99),
        }
        rows.append(row)
        slope_rows.append(row)
    summary = pd.DataFrame(rows)
    slopes = pd.DataFrame(slope_rows)
    if not summary.empty:
        summary = summary.sort_values(["source", "dataset", "stress_family", "relative_mse_auc", "model_order"])
    if not slopes.empty:
        slopes = slopes.sort_values(["source", "dataset", "stress_family", "degradation_slope", "model_order"])
    return summary, slopes


def build_structural_findings(
    severity_summary: pd.DataFrame,
    winners: pd.DataFrame,
    learned_winners: pd.DataFrame,
) -> pd.DataFrame:
    def slope(source: str, family: str, model: str) -> float:
        row = severity_summary[
            (severity_summary["source"] == source)
            & (severity_summary["stress_family"] == family)
            & (severity_summary["model"] == model)
        ]
        return float(row.iloc[0]["degradation_slope"]) if not row.empty else np.nan

    def p95(source: str, family: str, model: str) -> float:
        row = severity_summary[
            (severity_summary["source"] == source)
            & (severity_summary["stress_family"] == family)
            & (severity_summary["model"] == model)
        ]
        return float(row.iloc[0]["mean_p95_ms"]) if not row.empty else np.nan

    sf_dlinear_mv = slope("salesforce_borg", "missing_variables", "dlinear")
    sf_patch_mv = slope("salesforce_borg", "missing_variables", "patchtst")
    ali_dlinear_mv = slope("alibaba2018", "missing_variables", "dlinear")
    ali_patch_mv = slope("alibaba2018", "missing_variables", "patchtst")
    sf_noise = [
        slope("salesforce_borg", "noise", model)
        for model in ["dlinear", "race_dlinear", "patchtst"]
    ]
    ali_noise = [
        slope("alibaba2018", "noise", model)
        for model in ["dlinear", "race_dlinear", "patchtst"]
    ]
    sf_dlinear_p95 = p95("salesforce_borg", "missing_variables", "dlinear")
    sf_patch_p95 = p95("salesforce_borg", "missing_variables", "patchtst")
    latency_ratio = sf_patch_p95 / sf_dlinear_p95 if sf_dlinear_p95 and not np.isnan(sf_dlinear_p95) else np.nan

    selected = winners[
        (winners["source"] == "alibaba2018")
        & winners["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12", "level_shift"])
    ].copy()
    learned_selected = learned_winners[
        (learned_winners["source"] == "alibaba2018")
        & learned_winners["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12", "level_shift"])
    ].copy()
    mse_capacity_diff = int((selected["best_mse_model"] != selected["best_capacity_model"]).sum()) if not selected.empty else 0
    learned_mse_latency_diff = (
        int((learned_selected["best_mse_model"] != learned_selected["best_latency_model"]).sum())
        if not learned_selected.empty
        else 0
    )
    total = int(len(selected))
    learned_total = int(len(learned_selected))

    rows = [
        {
            "finding": "Metric-outage robustness is model-family specific",
            "evidence": (
                f"Salesforce/Borg missing-variable slope: DLinear {sf_dlinear_mv:.2f} vs PatchTST-lite {sf_patch_mv:.2f}; "
                f"Alibaba: {ali_dlinear_mv:.2f} vs {ali_patch_mv:.2f}."
            ),
            "deployment_implication": "Patch context helps when entire metric channels disappear, but it should be weighed against latency.",
        },
        {
            "finding": "Patch robustness is not universal",
            "evidence": (
                f"Noise slopes are close: Salesforce/Borg {min(sf_noise):.2f}-{max(sf_noise):.2f}, "
                f"Alibaba {min(ali_noise):.2f}-{max(ali_noise):.2f} for DLinear/RACE/PatchTST-lite."
            ),
            "deployment_implication": "A model that helps telemetry outage may not add much under point-wise sensor noise.",
        },
        {
            "finding": "Clean, latency, and capacity objectives disagree",
            "evidence": (
                f"On Alibaba core learned scenarios, best MSE differs from best learned P95 in "
                f"{learned_mse_latency_diff}/{learned_total} cases and from best capacity proxy in "
                f"{mse_capacity_diff}/{total} cases."
            ),
            "deployment_implication": "A clean leaderboard cannot decide between online alerting and proactive capacity planning.",
        },
        {
            "finding": "Robustness has measurable deployment cost",
            "evidence": (
                f"On Salesforce/Borg metric outage, PatchTST-lite has about {latency_ratio:.1f}x DLinear P95 latency "
                f"while roughly halving the degradation slope."
            ),
            "deployment_implication": "The benchmark should report Pareto choices, not only a single accuracy winner.",
        },
    ]
    return pd.DataFrame(rows)


def summarize_stressroute_policy(stressroute_v2: pd.DataFrame) -> pd.DataFrame:
    if stressroute_v2.empty:
        return pd.DataFrame()
    frame = stressroute_v2[
        (stressroute_v2["objective"] == "capacity")
        & (stressroute_v2["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12", "burst", "level_shift"]))
        & (stressroute_v2["latency_budget_ms"].isin([0.2, 0.5, 1.0]))
        & (stressroute_v2["policy"].isin(["fixed_patchtst", "stressroute_v1", "stressroute_v2", "oracle"]))
    ].copy()
    if frame.empty:
        return pd.DataFrame()

    def selection_summary(values: pd.Series) -> str:
        counts = values.dropna().astype(str).value_counts()
        if counts.empty:
            return ""
        return ", ".join(f"{model}:{count}" for model, count in counts.items())

    rows = []
    for (source, budget), group in frame.groupby(["source", "latency_budget_ms"], sort=True):
        row = {"source": source, "latency_budget_ms": budget}
        patch = group[group["policy"] == "fixed_patchtst"]
        v1 = group[group["policy"] == "stressroute_v1"]
        v2 = group[group["policy"] == "stressroute_v2"]
        oracle = group[group["policy"] == "oracle"]
        if not patch.empty:
            row["fixed_patchtst_feasible_rate"] = float(patch["budget_feasible"].astype(bool).mean())
            row["fixed_patchtst_mean_cost"] = float(patch["capacity_cost"].mean())
        if not v1.empty:
            row["v1_selected_models"] = selection_summary(v1["selected_model"])
            row["v1_mean_cost"] = float(v1["capacity_cost"].mean())
            row["v1_cost_vs_dlinear"] = float(v1["capacity_cost_vs_dlinear"].mean())
        if not v2.empty:
            row["v2_selected_models"] = selection_summary(v2["selected_model"])
            row["v2_mean_cost"] = float(v2["capacity_cost"].mean())
            row["v2_cost_vs_dlinear"] = float(v2["capacity_cost_vs_dlinear"].mean())
            row["v2_mean_p95_ms"] = float(v2["latency_p95_ms"].mean())
            row["v2_regret"] = float(v2["latency_constrained_regret"].mean())
            row["v2_oracle_gap"] = float(v2["capacity_oracle_gap"].mean())
        if not oracle.empty:
            row["oracle_mean_cost"] = float(oracle["capacity_cost"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_multiseed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    agg = (
        frame.groupby(["source", "dataset", "stress", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            mse_mean=("mse", "mean"),
            mse_std=("mse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            p95_ms_mean=("latency_p95_ms", "mean"),
            p95_ms_std=("latency_p95_ms", "std"),
            capacity_cost_mean=("capacity_cost", "mean"),
            capacity_cost_std=("capacity_cost", "std"),
            params=("params", "mean"),
            memory_mb=("max_memory_mb", "mean"),
        )
    )
    agg["model_order"] = agg["model"].map(MODEL_ORDER).fillna(99)
    return agg.sort_values(["source", "dataset", "stress", "mse_mean", "model_order"]).drop(columns=["model_order"])


def compact_multiseed_summary(multiseed_summary: pd.DataFrame) -> pd.DataFrame:
    if multiseed_summary.empty:
        return pd.DataFrame()
    selected = multiseed_summary[
        (multiseed_summary["source"].isin(["alibaba2018", "salesforce_borg"]))
        & (multiseed_summary["stress"].isin(["clean", "missing_variables_30"]))
        & (multiseed_summary["model"].isin(["dlinear", "patchtst"]))
    ].copy()
    if selected.empty:
        return selected
    selected["stress_order"] = selected["stress"].map({"clean": 0, "missing_variables_30": 1}).fillna(99)
    selected["model_order"] = selected["model"].map(MODEL_ORDER).fillna(99)
    keep = [
        "source",
        "dataset",
        "stress",
        "model",
        "seeds",
        "mse_mean",
        "mse_std",
        "capacity_cost_mean",
        "capacity_cost_std",
        "p95_ms_mean",
        "p95_ms_std",
    ]
    return selected.sort_values(["source", "stress_order", "model_order"])[keep]


def compact_five_seed_winners(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {
        "pool",
        "dataset",
        "stress",
        "min_seeds",
        "best_mse_model",
        "best_capacity_model",
        "lowest_latency_model",
        "mse_vs_capacity_disagreement",
    }
    if not required.issubset(frame.columns):
        return pd.DataFrame()
    selected = frame[
        (frame["pool"] == "core_lightweight")
        & (frame["dataset"].isin(["alibaba2018", "salesforce_borg"]))
    ].copy()
    if selected.empty:
        return selected
    paired_path = path.with_name("five_seed_paired_tests.csv")
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        paired_cols = ["dataset", "stress", "paired_t_p"]
        if set(paired_cols).issubset(paired.columns):
            selected = selected.merge(paired[paired_cols], on=["dataset", "stress"], how="left")
        else:
            selected["paired_t_p"] = np.nan
    else:
        selected["paired_t_p"] = np.nan
    selected["seeds"] = selected["min_seeds"].astype(int)
    selected["mse_vs_capacity_flip"] = selected["mse_vs_capacity_disagreement"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
    )
    selected["source_order"] = selected["dataset"].map({"alibaba2018": 0, "salesforce_borg": 1}).fillna(99)
    selected["stress_order"] = selected["stress"].map(
        {"clean": 0, "delayed_12": 1, "level_shift": 2, "missing_30": 3, "missing_variables_30": 4}
    ).fillna(99)
    keep = [
        "dataset",
        "stress",
        "seeds",
        "best_mse_model",
        "best_capacity_model",
        "lowest_latency_model",
        "mse_vs_capacity_flip",
        "paired_t_p",
    ]
    return selected.sort_values(["source_order", "stress_order"])[keep]


def summarize_cost_ratio_sensitivity(capacity_sim: pd.DataFrame) -> pd.DataFrame:
    if capacity_sim.empty:
        return pd.DataFrame()
    frame = capacity_sim[
        (capacity_sim["policy"] == "forecast_capacity")
        & (capacity_sim["source"].isin(["alibaba2018", "salesforce_borg"]))
        & (capacity_sim["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12"]))
        & (capacity_sim["model"].isin(["dlinear", "race_dlinear", "patchtst"]))
    ].copy()
    if frame.empty:
        return frame
    rows = []
    for ratio, under_cost in [("2:1", 2.0), ("5:1", 5.0), ("10:1", 10.0)]:
        frame[f"cost_ratio_{ratio}"] = under_cost * frame["under_provision_area"] + frame["over_provision_area"]
        for keys, group in frame.groupby(["source", "stress"], sort=True):
            source, stress = keys
            summary = (
                group.groupby("model", as_index=False)
                .agg(
                    total_cost=(f"cost_ratio_{ratio}", "mean"),
                    under_area=("under_provision_area", "mean"),
                    over_area=("over_provision_area", "mean"),
                    peak_miss=("peak_miss_rate", "mean"),
                    p95_ms=("latency_p95_ms", "mean"),
                )
                .sort_values(["total_cost", "model"], kind="stable")
            )
            if summary.empty:
                continue
            winner = summary.iloc[0]
            rows.append(
                {
                    "source": source,
                    "stress": stress,
                    "cost_ratio": ratio,
                    "under_cost": under_cost,
                    "over_cost": 1.0,
                    "best_total_cost_model": winner["model"],
                    "best_total_cost": winner["total_cost"],
                    "best_under_area": winner["under_area"],
                    "best_over_area": winner["over_area"],
                    "best_peak_miss": winner["peak_miss"],
                    "best_p95_ms": winner["p95_ms"],
                }
            )
    return pd.DataFrame(rows)


def summarize_imputation_pipeline(imputation: pd.DataFrame) -> pd.DataFrame:
    if imputation.empty:
        return imputation
    frame = imputation[
        (imputation["source"].isin(["alibaba2018", "salesforce_borg"]))
        & (imputation["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12", "level_shift"]))
        & (imputation["model"].isin(["dlinear", "patchtst"]))
        & (imputation["imputation"].isin(["none", "ffill", "mean"]))
    ].copy()
    if frame.empty:
        return frame
    frame["stress_order"] = frame["stress"].map(
        {"clean": 0, "missing_30": 1, "missing_variables_30": 2, "delayed_12": 3, "level_shift": 4}
    ).fillna(99)
    frame["model_order"] = frame["model"].map(MODEL_ORDER).fillna(99)
    keep = [
        "source",
        "stress",
        "model",
        "imputation",
        "pipeline",
        "mse",
        "mae",
        "latency_p95_ms",
        "capacity_cost",
    ]
    return frame.sort_values(["source", "stress_order", "model_order", "imputation"])[keep]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready summary tables from RACE-Forecast runs.")
    parser.add_argument("--gaia-matrix", default="outputs/gaia_matrix_core_summary.csv")
    parser.add_argument("--netman-summary", default="outputs/netman_kpi_core_summary.csv")
    parser.add_argument("--alibaba-summary", default="outputs/alibaba2018_machine_usage_core_summary.csv")
    parser.add_argument("--metrics-summary", action="append", default=[], help="Extra native summary in source=path format.")
    parser.add_argument("--ablation", default="outputs/gaia_ablation_summary.csv")
    parser.add_argument("--official-dir", default="outputs/official_baselines")
    parser.add_argument(
        "--official-patchtst",
        action="append",
        default=None,
        help="Official PatchTST native summary CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--official-itransformer-native",
        action="append",
        default=None,
        help="Official iTransformer native summary CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--official-timemixer",
        action="append",
        default=None,
        help="Official TimeMixer native summary CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--chronos-reference",
        action="append",
        default=None,
        help="Chronos zero-shot reference summary CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--capacity-simulator",
        action="append",
        default=None,
        help="Forecast-to-capacity simulator CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--capacity-simulator-sensitivity",
        action="append",
        default=None,
        help="Forecast-to-capacity sensitivity CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--severity",
        action="append",
        default=None,
        help="Native severity summary CSV. Can be provided multiple times.",
    )
    parser.add_argument("--official-severity", action="append", default=[], help="Optional official native severity summary CSV.")
    parser.add_argument(
        "--multiseed",
        action="append",
        default=None,
        help="Multi-seed summary CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--five-seed-winners",
        default="outputs/five_seed/main_five_seed_winners.csv",
        help="Five-seed winner summary CSV used for the main compact stability table.",
    )
    parser.add_argument(
        "--stressroute",
        action="append",
        default=None,
        help="StressRoute report CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--stressroute-v2",
        action="append",
        default=None,
        help="StressRoute v2 policy comparison CSV. Can be provided multiple times.",
    )
    parser.add_argument(
        "--imputation",
        action="append",
        default=None,
        help="Imputation + forecasting pipeline summary CSV. Can be provided multiple times.",
    )
    parser.add_argument("--output-dir", default="outputs/paper_tables")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset audit item in name=path format.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if Path(args.gaia_matrix).exists():
        frames.append(load_metrics(Path(args.gaia_matrix), "gaia"))
    if Path(args.netman_summary).exists():
        frames.append(load_metrics(Path(args.netman_summary), "netman"))
    if Path(args.alibaba_summary).exists():
        frames.append(load_metrics(Path(args.alibaba_summary), "alibaba2018"))
    for item in args.metrics_summary:
        if "=" not in item:
            raise ValueError(f"Extra summary must use source=path, got {item}")
        source, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if path.exists():
            frames.append(load_metrics(path, source))
        else:
            print(f"Warning: skip missing extra summary path: {path}")
    official = load_official_metrics(Path(args.official_dir))
    if not official.empty:
        frames.append(official)
    official_patchtst_paths = args.official_patchtst or ["outputs/official_patchtst_native_summary.csv"]
    for official_patchtst_path in official_patchtst_paths:
        official_patchtst = load_official_patchtst(Path(official_patchtst_path))
        if not official_patchtst.empty:
            frames.append(official_patchtst)
    native_itransformer_paths = args.official_itransformer_native or ["outputs/official_itransformer_native_summary.csv"]
    for native_itransformer_path in native_itransformer_paths:
        native_itransformer = load_native_official(Path(native_itransformer_path), "official_itransformer_native")
        if not native_itransformer.empty:
            frames.append(native_itransformer)
    timemixer_paths = args.official_timemixer or []
    for timemixer_path in timemixer_paths:
        timemixer = load_official_timemixer(Path(timemixer_path))
        if not timemixer.empty:
            frames.append(timemixer)
    chronos_paths = args.chronos_reference or []
    for chronos_path in chronos_paths:
        chronos_ref = load_native_official(Path(chronos_path), "chronos_bolt_reference")
        if not chronos_ref.empty:
            frames.append(chronos_ref)
    if not frames:
        raise SystemExit("No metrics files found.")
    combined = pd.concat(frames, ignore_index=True)
    capacity_sim_frames = []
    capacity_sim_paths = args.capacity_simulator or ["outputs/capacity_simulator_summary.csv"]
    for capacity_path in capacity_sim_paths:
        frame = load_capacity_simulator(Path(capacity_path))
        if not frame.empty:
            capacity_sim_frames.append(frame)
    capacity_sim = pd.concat(capacity_sim_frames, ignore_index=True) if capacity_sim_frames else pd.DataFrame()
    if not capacity_sim.empty:
        dedupe_cols = [
            col
            for col in ["source", "dataset", "stress", "model", "policy", "pred_len", "headroom"]
            if col in capacity_sim.columns
        ]
        capacity_sim = capacity_sim.drop_duplicates(subset=dedupe_cols, keep="last")
    capacity_sensitivity_frames = []
    capacity_sensitivity_paths = args.capacity_simulator_sensitivity or ["outputs/capacity_simulator_sensitivity_summary.csv"]
    for sensitivity_path in capacity_sensitivity_paths:
        frame = load_capacity_simulator(Path(sensitivity_path))
        if not frame.empty:
            capacity_sensitivity_frames.append(frame)
    capacity_sensitivity = pd.concat(capacity_sensitivity_frames, ignore_index=True) if capacity_sensitivity_frames else pd.DataFrame()
    if not capacity_sensitivity.empty:
        dedupe_cols = [
            col
            for col in ["source", "dataset", "stress", "model", "policy", "pred_len", "headroom"]
            if col in capacity_sensitivity.columns
        ]
        capacity_sensitivity = capacity_sensitivity.drop_duplicates(subset=dedupe_cols, keep="last")
    severity_frames = []
    severity_paths = args.severity or ["outputs/severity_curve_summary.csv"]
    for severity_path in severity_paths:
        severity = load_severity(Path(severity_path))
        if not severity.empty:
            severity_frames.append(severity)
    for severity_path in args.official_severity:
        official_severity = load_severity(Path(severity_path))
        if not official_severity.empty:
            severity_frames.append(official_severity)
    severity = pd.concat(severity_frames, ignore_index=True) if severity_frames else pd.DataFrame()
    multiseed_frames = []
    multiseed_paths = args.multiseed or ["outputs/multiseed_summary.csv"]
    for multiseed_path in multiseed_paths:
        seed_frame = load_multiseed(Path(multiseed_path))
        if not seed_frame.empty:
            multiseed_frames.append(seed_frame)
    multiseed = pd.concat(multiseed_frames, ignore_index=True) if multiseed_frames else pd.DataFrame()
    stressroute_frames = []
    stressroute_paths = args.stressroute or ["outputs/stressroute_v1_alibaba_patchtst_report.csv"]
    for stressroute_path in stressroute_paths:
        route_frame = load_stressroute(Path(stressroute_path))
        if not route_frame.empty:
            stressroute_frames.append(route_frame)
    stressroute = pd.concat(stressroute_frames, ignore_index=True) if stressroute_frames else pd.DataFrame()
    stressroute_v2_frames = []
    for stressroute_path in args.stressroute_v2 or []:
        path = Path(stressroute_path)
        if path.exists():
            stressroute_v2_frames.append(pd.read_csv(path))
    stressroute_v2 = pd.concat(stressroute_v2_frames, ignore_index=True) if stressroute_v2_frames else pd.DataFrame()
    imputation_frames = []
    for imputation_path in args.imputation or []:
        path = Path(imputation_path)
        if path.exists():
            imputation_frames.append(pd.read_csv(path))
        else:
            print(f"Warning: skip missing imputation path: {path}")
    imputation = pd.concat(imputation_frames, ignore_index=True) if imputation_frames else pd.DataFrame()
    for col in [
        "mse",
        "mae",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_mean_ms",
        "capacity_under_rate",
        "capacity_over_rate",
        "capacity_mean_under",
        "capacity_mean_over",
        "capacity_cost",
        "params",
        "max_memory_mb",
        "train_seconds",
        "eval_seconds",
    ]:
        if col not in combined.columns:
            combined[col] = np.nan
    core_combined = main_comparison_frame(combined)

    write_table(build_benchmark_card(), output_dir, "table_benchmark_card")
    write_table(build_baseline_fairness(combined), output_dir, "table_baseline_fairness")

    metric_cols = ["mse", "mae", "latency_p95_ms", "capacity_cost", "params", "max_memory_mb"]
    clean = (
        core_combined[core_combined["stress"] == "clean"]
        .groupby(["source", "model"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["source", "mse", "latency_p95_ms"])
    )
    write_table(clean, output_dir, "table_clean_accuracy")

    cost = (
        core_combined.groupby(["source", "model"], as_index=False)
        .agg(
            clean_p95_ms=("latency_p95_ms", lambda x: float(x[combined.loc[x.index, "stress"].eq("clean")].mean())),
            mean_p95_ms=("latency_p95_ms", "mean"),
            mean_p50_ms=("latency_p50_ms", "mean"),
            params=("params", "mean"),
            memory_mb=("max_memory_mb", "mean"),
            train_seconds=("train_seconds", "mean"),
            eval_seconds=("eval_seconds", "mean"),
        )
        .sort_values(["source", "mean_p95_ms", "params"])
    )
    write_table(cost, output_dir, "table_deployment_cost")

    stress = (
        core_combined[core_combined["stress"] != "clean"]
        .groupby(["source", "model"], as_index=False)
        .agg(
            mean_mse=("mse", "mean"),
            mean_mae=("mae", "mean"),
            worst_mse=("mse", "max"),
            mean_p95_ms=("latency_p95_ms", "mean"),
            mean_capacity_cost=("capacity_cost", "mean"),
            params=("params", "mean"),
            memory_mb=("max_memory_mb", "mean"),
        )
        .sort_values(["source", "mean_mse", "mean_p95_ms"])
    )
    write_table(stress, output_dir, "table_stress_robustness")

    risk = (
        core_combined[
            core_combined["source"].isin(["alibaba2018", "netman"])
            & core_combined["stress"].ne("clean")
            & core_combined["capacity_cost"].notna()
        ]
        .groupby(["source", "model"], as_index=False)
        .agg(
            mean_under_rate=("capacity_under_rate", "mean"),
            mean_over_rate=("capacity_over_rate", "mean"),
            mean_capacity_cost=("capacity_cost", "mean"),
            worst_capacity_cost=("capacity_cost", "max"),
            mean_p95_ms=("latency_p95_ms", "mean"),
            params=("params", "mean"),
        )
        .sort_values(["source", "mean_capacity_cost", "mean_p95_ms"])
    )
    write_table(risk, output_dir, "table_capacity_risk_proxy")

    if not capacity_sim.empty:
        primary_pred_len = 24.0 if (capacity_sim["pred_len"].astype(float) == 24.0).any() else np.nan
        primary_headroom = 0.15 if (capacity_sim["headroom"].astype(float) == 0.15).any() else np.nan
        if pd.isna(primary_pred_len):
            pred_mode = capacity_sim["pred_len"].dropna().mode()
            primary_pred_len = float(pred_mode.iloc[0]) if not pred_mode.empty else np.nan
        if pd.isna(primary_headroom):
            headroom_mode = capacity_sim["headroom"].dropna().mode()
            primary_headroom = float(headroom_mode.iloc[0]) if not headroom_mode.empty else np.nan
        primary_sim = capacity_sim.copy()
        if not pd.isna(primary_pred_len):
            primary_sim = primary_sim[primary_sim["pred_len"].astype(float) == primary_pred_len]
        if not pd.isna(primary_headroom):
            primary_sim = primary_sim[primary_sim["headroom"].astype(float) == primary_headroom]
        sim_forecast = capacity_sim[capacity_sim["policy"] == "forecast_capacity"].copy()
        sim_reactive = capacity_sim[capacity_sim["policy"] == "reactive_baseline"].copy()
        sim_summary = (
            primary_sim[primary_sim["stress"].ne("clean")]
            .groupby(["source", "model", "policy"], as_index=False)
            .agg(
                mean_total_cost=("total_normalized_cost", "mean"),
                worst_total_cost=("total_normalized_cost", "max"),
                mean_under_area=("under_provision_area", "mean"),
                mean_over_area=("over_provision_area", "mean"),
                mean_peak_miss=("peak_miss_rate", "mean"),
                mean_p95_under=("p95_under_ratio", "mean"),
                latency_p95_ms=("latency_p95_ms", "mean"),
                params=("params", "mean"),
            )
            .sort_values(["source", "mean_total_cost", "latency_p95_ms"])
        )
        write_table(sim_summary, output_dir, "table_capacity_simulator")

        sim_rows = []
        forecast_only = primary_sim[(primary_sim["stress"].ne("clean")) & (primary_sim["policy"] == "forecast_capacity")]
        for keys, group in forecast_only.groupby(["source", "stress"], sort=True):
            source, stress_name = keys
            row = {"source": source, "stress": stress_name}
            for metric, label in [
                ("total_normalized_cost", "best_total_cost_model"),
                ("under_provision_area", "best_under_area_model"),
                ("peak_miss_rate", "best_peak_miss_model"),
                ("latency_p95_ms", "best_latency_model"),
            ]:
                usable = group.dropna(subset=[metric]).copy()
                if usable.empty:
                    continue
                winner = usable.sort_values([metric, "model_order"]).iloc[0]
                row[label] = winner["model"]
                row[f"{metric}_best"] = winner[metric]
            sim_rows.append(row)
        write_table(pd.DataFrame(sim_rows), output_dir, "table_capacity_simulator_winners")

        sensitivity_frame = capacity_sensitivity if not capacity_sensitivity.empty else capacity_sim
        if "pred_len" not in sensitivity_frame.columns:
            sensitivity_frame["pred_len"] = primary_pred_len
        if "headroom" not in sensitivity_frame.columns:
            sensitivity_frame["headroom"] = primary_headroom

        horizon = (
            sensitivity_frame[(sensitivity_frame["policy"] == "forecast_capacity") & sensitivity_frame["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12"])]
            .groupby(["source", "stress", "model", "pred_len"], as_index=False)
            .agg(
                total_cost=("total_normalized_cost", "mean"),
                under_area=("under_provision_area", "mean"),
                over_area=("over_provision_area", "mean"),
                peak_miss=("peak_miss_rate", "mean"),
                p95_ms=("latency_p95_ms", "mean"),
            )
            .sort_values(["source", "stress", "pred_len", "total_cost"])
        )
        write_table(horizon, output_dir, "table_capacity_horizon_sensitivity")

        headroom = (
            sensitivity_frame[(sensitivity_frame["policy"] == "forecast_capacity") & sensitivity_frame["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12"])]
            .groupby(["source", "stress", "model", "headroom"], as_index=False)
            .agg(
                total_cost=("total_normalized_cost", "mean"),
                under_area=("under_provision_area", "mean"),
                over_area=("over_provision_area", "mean"),
                peak_miss=("peak_miss_rate", "mean"),
                p95_ms=("latency_p95_ms", "mean"),
            )
            .sort_values(["source", "stress", "headroom", "total_cost"])
        )
        write_table(headroom, output_dir, "table_capacity_headroom_sensitivity")

    severity_summary, severity_slopes = summarize_severity(severity)
    if not severity_summary.empty:
        write_table(severity_summary, output_dir, "table_severity_auc")
        write_table(severity_slopes, output_dir, "table_severity_slope")

    multiseed_summary = summarize_multiseed(multiseed)
    if not multiseed_summary.empty:
        write_table(multiseed_summary, output_dir, "table_multiseed_stability")
    compact_five_seed = compact_five_seed_winners(Path(args.five_seed_winners))
    if not compact_five_seed.empty:
        write_table(compact_five_seed, output_dir, "table_multiseed_compact")
    elif not multiseed_summary.empty:
        compact_multiseed = compact_multiseed_summary(multiseed_summary)
        if not compact_multiseed.empty:
            write_table(compact_multiseed, output_dir, "table_multiseed_compact")

    cost_ratio_sensitivity = summarize_cost_ratio_sensitivity(capacity_sim)
    if not cost_ratio_sensitivity.empty:
        write_table(cost_ratio_sensitivity, output_dir, "table_capacity_cost_ratio_sensitivity")

    if not stressroute.empty:
        selected_route = stressroute[
            (stressroute["policy"] == "stressroute_v1_capacity")
            & (stressroute["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12", "burst", "level_shift"]))
        ].copy()
        selected_route = selected_route.sort_values(["source", "stress", "latency_budget_ms", "selected_model_order"])
        keep_cols = [
            "source",
            "dataset",
            "stress",
            "latency_budget_ms",
            "selected_model",
            "route_reason",
            "mse",
            "mae",
            "capacity_cost",
            "latency_p95_ms",
            "mse_vs_dlinear",
            "capacity_cost_vs_dlinear",
            "latency_vs_dlinear",
        ]
        write_table(selected_route[keep_cols], output_dir, "table_stressroute_v1")

    if not stressroute_v2.empty:
        selected = stressroute_v2[
            (stressroute_v2["objective"] == "capacity")
            & (stressroute_v2["stress"].isin(["clean", "missing_30", "missing_variables_30", "delayed_12", "burst", "level_shift"]))
            & (stressroute_v2["latency_budget_ms"].isin([0.2, 0.5, 1.0]))
            & (stressroute_v2["policy"].isin(["fixed_dlinear", "fixed_patchtst", "stressroute_v1", "stressroute_v2", "oracle"]))
        ].copy()
        if not selected.empty:
            selected["policy_order"] = selected["policy"].map(
                {
                    "fixed_dlinear": 0,
                    "fixed_patchtst": 1,
                    "stressroute_v1": 2,
                    "stressroute_v2": 3,
                    "oracle": 4,
                }
            ).fillna(99)
            selected = selected.sort_values(["source", "stress", "latency_budget_ms", "policy_order"])
            keep_cols = [
                "source",
                "dataset",
                "stress",
                "latency_budget_ms",
                "policy",
                "selected_model",
                "mse",
                "capacity_cost",
                "latency_p95_ms",
                "budget_feasible",
                "route_model_count",
                "capacity_cost_vs_dlinear",
                "latency_vs_dlinear",
                "capacity_cost_vs_patchtst",
                "latency_vs_patchtst",
                "latency_constrained_regret",
                "latency_constrained_regret_ratio",
                "capacity_oracle_gap",
            ]
            keep_cols = [col for col in keep_cols if col in selected.columns]
            write_table(selected[keep_cols], output_dir, "table_stressroute_v2")

    all_available_winners = add_best_columns(core_combined, ["source", "stress"])
    write_table(all_available_winners, output_dir, "table_scenario_winners_all_available")

    core_learned_combined = core_learned_frame(core_combined)
    winners = add_best_columns(core_learned_combined, ["source", "stress"])
    write_table(winners, output_dir, "table_scenario_winners")

    if not severity_summary.empty and not winners.empty:
        write_table(
            build_structural_findings(severity_summary, winners, winners),
            output_dir,
            "table_structural_findings",
        )

    win_rates = summarize_win_rates(core_combined)
    write_table(win_rates, output_dir, "table_win_counts")

    stressroute_policy = summarize_stressroute_policy(stressroute_v2)
    if not stressroute_policy.empty:
        write_table(stressroute_policy, output_dir, "table_stressroute_policy_summary")

    imputation_summary = summarize_imputation_pipeline(imputation)
    if not imputation_summary.empty:
        write_table(imputation_summary, output_dir, "table_imputation_pipeline")

    write_table(build_deployment_guidelines(), output_dir, "table_deployment_guidelines")

    gaia_category = add_best_columns(core_combined[core_combined["source"] == "gaia"], ["dataset", "stress"])
    write_table(gaia_category, output_dir, "table_gaia_category_analysis")

    ablation_path = Path(args.ablation)
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        keep = ["run", "model", "mse", "mae", "latency_p95_ms", "capacity_cost", "params", "max_memory_mb"]
        write_table(ablation[keep].sort_values(["run", "mse"]), output_dir, "table_race_ablation")

    audit_rows = []
    for item in args.dataset:
        if "=" not in item:
            raise ValueError(f"Dataset audit must use name=path, got {item}")
        name, raw_path = item.split("=", 1)
        dataset_path = Path(raw_path)
        if not dataset_path.exists():
            print(f"Warning: skip missing dataset audit path: {dataset_path}")
            continue
        audit_rows.append(audit_npz(name, dataset_path))
    if audit_rows:
        write_table(pd.DataFrame(audit_rows), output_dir, "table_dataset_audit")

    narrative = [
        "# Paper Table Notes",
        "",
        "- `table_clean_accuracy` reports clean forecasting quality and deployment cost averaged by data source.",
        "- `table_stress_robustness` excludes clean data and summarizes robustness across missing, delayed, burst, and shift scenarios.",
        "- `table_deployment_cost` summarizes P50/P95 latency, parameters, memory, and official-baseline runtime when available.",
        "- `table_capacity_risk_proxy` reports operational under/over-provision proxy only for Alibaba 2018 and NetMan, where it is more defensible than on GAIA.",
        "- `table_capacity_simulator` reports forecast-to-capacity policy outcomes, including reactive baselines, under/over area, peak miss, and total normalized cost.",
        "- `table_capacity_horizon_sensitivity` and `table_capacity_headroom_sensitivity` report deployment sensitivity of proactive forecast-capacity policies.",
        "- `table_severity_auc` and `table_severity_slope` summarize robustness degradation curves across stress intensities.",
        "- `table_multiseed_stability` reports seed-level stability for Alibaba and Salesforce/Borg core scenarios.",
        "- `table_multiseed_compact` is the main-text five-seed winner summary over 10 source-stress settings.",
        "- `table_capacity_cost_ratio_sensitivity` recomputes capacity winners under 2:1, 5:1, and 10:1 under/over cost ratios without retraining models.",
        "- `table_stress_realism_audit` audits natural non-finite, zero-run, flatline, spike, and level-shift signals in public telemetry.",
        "- `table_imputation_pipeline` reports a compact DLinear/PatchTST-lite imputation check under the native stress protocol.",
        "- `table_stressroute_v1` reports the interpretable stress-aware deployment policy under latency budgets.",
        "- `table_stressroute_v2` compares fixed models, StressRoute v1, StressRoute v2, and oracle selection when the v2 CSVs are available.",
        "- `table_stressroute_policy_summary` compresses the mixed-stress routing results for main-paper discussion.",
        "- `table_benchmark_card` and `table_baseline_fairness` are reviewer-facing cards for protocol and baseline comparability.",
        "- `table_structural_findings` extracts quantitative findings from `table_severity_auc`, `table_severity_slope`, and winner tables; do not hand-edit these claims.",
        "- `table_scenario_winners` is the main tradeoff table over the core learned comparable pool; LastValue, Chronos-Bolt, and LTSF-bridge references are excluded.",
        "- `table_scenario_winners_all_available` is retained as an artifact-only reference that includes all available rows.",
        "- `table_deployment_guidelines` translates the benchmark evidence into deployment-model selection guidance.",
        "- `table_gaia_category_analysis` checks whether conclusions hold across GAIA periodic, changepoint, low-SNR, and partially-stationary subsets.",
        "- Alibaba 2018 machine usage is the main multivariate CloudOps source for CPU/memory/network/disk stress evaluation.",
        "- `official_patchtst` uses the official PatchTST model class inside the native stress pipeline; `patchtst` is the lightweight PatchTST-lite baseline.",
        "- `official_itransformer_native` uses the official iTransformer model class inside the native stress pipeline; the older `official_itransformer` rows come from the LTSF CSV bridge.",
        "- `official_timemixer` uses THUML Time-Series-Library TimeMixer inside the native stress pipeline.",
        "- `table_race_ablation` should be used to discuss RACE-DLinear as a lightweight robust baseline, not as a universal leading model.",
    ]
    (output_dir / "README.md").write_text("\n".join(narrative) + "\n", encoding="utf-8")
    print(f"Saved {output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
