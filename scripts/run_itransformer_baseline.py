from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DATASETS = {
    "gaia": {
        "csv": "gaia_metric_ltsf.csv",
        "enc_in": 64,
    },
    "netman": {
        "csv": "netman_kpi_ltsf.csv",
        "enc_in": 29,
    },
    "alibaba2018": {
        "csv": "alibaba2018_machine_usage_ltsf.csv",
        "enc_in": 320,
    },
}

STRESSES = {
    "clean": {},
    "missing_points_30": {"--eval_time_mask_ratio": "0.3"},
    "missing_variables_30": {"--eval_channel_mask_ratio": "0.3"},
    "noise_20": {"--eval_noise_std": "0.2"},
}


def read_target(csv_path: Path) -> str:
    columns = pd.read_csv(csv_path, nrows=1).columns.tolist()
    if len(columns) < 2:
        raise ValueError(f"LTSF CSV must contain date and at least one variable: {csv_path}")
    return columns[-1]


def setting_name(
    model_id: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    d_model: int,
    n_heads: int,
    e_layers: int,
    d_layers: int,
    d_ff: int,
    factor: int,
    des: str,
) -> str:
    return (
        f"{model_id}_iTransformer_custom_M_ft{seq_len}_sl{label_len}_ll{pred_len}_pl"
        f"{d_model}_dm{n_heads}_nh{e_layers}_el{d_layers}_dl{d_ff}_df{factor}_fc"
        f"timeF_ebTrue_dt{des}_projection_0"
    )


def stress_setting(base_setting: str, stress: str) -> str:
    if stress == "clean":
        return base_setting
    tags = {
        "missing_points_30": "timemask0.3",
        "missing_variables_30": "chmask0.3",
        "noise_20": "noise0.2",
    }
    return f"{base_setting}_stress_{tags[stress]}"


def count_params(repo: Path, python: str, args: list[str]) -> int:
    script = """
import argparse
from model import iTransformer
parser = argparse.ArgumentParser()
parser.add_argument('--seq_len', type=int)
parser.add_argument('--pred_len', type=int)
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--use_norm', type=int, default=True)
parser.add_argument('--d_model', type=int)
parser.add_argument('--embed', default='timeF')
parser.add_argument('--freq', default='t')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--class_strategy', default='projection')
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--n_heads', type=int)
parser.add_argument('--e_layers', type=int)
parser.add_argument('--d_ff', type=int)
parser.add_argument('--activation', default='gelu')
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--use_dasr', action='store_true')
cfg = parser.parse_args()
print(sum(p.numel() for p in iTransformer.Model(cfg).parameters()))
"""
    proc = subprocess.run(
        [python, "-c", script, *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return int(proc.stdout.strip().splitlines()[-1])


def run_command(cmd: list[str], cwd: Path, env: dict[str, str]) -> tuple[float, str]:
    start = time.time()
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    elapsed = time.time() - start
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd, proc.stdout, proc.stderr)
    return elapsed, proc.stdout


def parse_metrics(repo: Path, result_setting: str) -> tuple[float, float]:
    metrics_path = repo / "results" / result_setting / "metrics.npy"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    mae, mse, *_ = np.load(metrics_path)
    return float(mse), float(mae)


def copy_outputs(repo: Path, result_setting: str, output_dir: Path, dataset: str, stress: str) -> None:
    target = output_dir / "raw" / dataset / stress
    target.mkdir(parents=True, exist_ok=True)
    for subdir in ["results", "test_results"]:
        src = repo / subdir / result_setting
        if src.exists():
            dst = target / subdir
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def sync_checkpoint_for_eval(repo: Path, output_dir: Path, setting: str) -> None:
    saved = output_dir / "checkpoints" / setting / "checkpoint.pth"
    if not saved.exists():
        raise FileNotFoundError(saved)
    expected_dir = repo / "checkpoints" / setting
    expected_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(saved, expected_dir / "checkpoint.pth")


def profile_latency(
    repo: Path,
    python: str,
    checkpoint: Path,
    model_args: list[str],
    enc_in: int,
    iterations: int,
    warmup: int,
    gpu: str,
) -> dict[str, float]:
    script = """
import argparse
import json
import os
import time
import numpy as np
import torch
from model import iTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--seq_len', type=int, required=True)
parser.add_argument('--pred_len', type=int, required=True)
parser.add_argument('--enc_in', type=int, required=True)
parser.add_argument('--d_model', type=int, required=True)
parser.add_argument('--n_heads', type=int, required=True)
parser.add_argument('--e_layers', type=int, required=True)
parser.add_argument('--d_layers', type=int, required=True)
parser.add_argument('--d_ff', type=int, required=True)
parser.add_argument('--iters', type=int, required=True)
parser.add_argument('--warmup', type=int, required=True)
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--use_norm', type=int, default=True)
parser.add_argument('--embed', default='timeF')
parser.add_argument('--freq', default='t')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--class_strategy', default='projection')
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--activation', default='gelu')
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--use_dasr', action='store_true')
cfg = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = iTransformer.Model(cfg).float().to(device)
model.load_state_dict(torch.load(cfg.checkpoint, map_location=device))
model.eval()
x = torch.randn(1, cfg.seq_len, cfg.enc_in, device=device)
x_mark = torch.randn(1, cfg.seq_len, 5, device=device)
y = torch.zeros(1, cfg.pred_len, cfg.enc_in, device=device)
y_mark = torch.randn(1, cfg.pred_len, 5, device=device)
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats(device)
times = []
with torch.no_grad():
    for _ in range(cfg.warmup):
        _ = model(x, x_mark, y, y_mark)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    for _ in range(cfg.iters):
        start = time.perf_counter()
        _ = model(x, x_mark, y, y_mark)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
peak = torch.cuda.max_memory_allocated(device) / 1048576 if device.type == 'cuda' else 0
print(json.dumps({
    'latency_p50_ms': float(np.percentile(times, 50)),
    'latency_p95_ms': float(np.percentile(times, 95)),
    'latency_mean_ms': float(np.mean(times)),
    'max_memory_mb': float(peak),
}))
"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    proc = subprocess.run(
        [python, "-c", script, "--checkpoint", str(checkpoint), "--enc_in", str(enc_in), *model_args, "--iters", str(iterations), "--warmup", str(warmup)],
        cwd=repo,
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    import json

    return json.loads(proc.stdout.strip().splitlines()[-1])


def read_existing(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run official iTransformer baseline on exported LTSF CSV files.")
    parser.add_argument("--repo", default="external/iTransformer_official_run")
    parser.add_argument("--python", default=".conda-env/bin/python")
    parser.add_argument("--data-root", default="data/ltsf")
    parser.add_argument("--output-dir", default="outputs/official_baselines")
    parser.add_argument("--datasets", nargs="*", default=["gaia", "netman"])
    parser.add_argument("--stresses", nargs="*", default=["clean", "missing_points_30", "missing_variables_30", "noise_20"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=24)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--e-layers", type=int, default=2)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--latency-iters", type=int, default=40)
    parser.add_argument("--latency-warmup", type=int, default=8)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    python = str(Path(args.python).resolve())
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    clean_path = output_dir / "itransformer_clean.csv"
    stress_path = output_dir / "itransformer_stress.csv"
    clean_rows = read_existing(clean_path) if args.append else []
    stress_rows = read_existing(stress_path) if args.append else []
    for dataset in args.datasets:
        meta = DATASETS[dataset]
        csv_path = data_root / meta["csv"]
        target = read_target(csv_path)
        model_id = f"{dataset}_itransformer_96_24"
        des = "official"
        base_setting = setting_name(
            model_id,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            1,
            des,
        )
        common = [
            "--model_id", model_id,
            "--model", "iTransformer",
            "--data", "custom",
            "--root_path", str(data_root) + "/",
            "--data_path", meta["csv"],
            "--features", "M",
            "--target", target,
            "--freq", "t",
            "--seq_len", str(args.seq_len),
            "--label_len", str(args.label_len),
            "--pred_len", str(args.pred_len),
            "--enc_in", str(meta["enc_in"]),
            "--dec_in", str(meta["enc_in"]),
            "--c_out", str(meta["enc_in"]),
            "--d_model", str(args.d_model),
            "--n_heads", str(args.n_heads),
            "--e_layers", str(args.e_layers),
            "--d_layers", str(args.d_layers),
            "--d_ff", str(args.d_ff),
            "--batch_size", str(args.batch_size),
            "--train_epochs", str(args.epochs),
            "--patience", "3",
            "--learning_rate", str(args.lr),
            "--num_workers", "0",
            "--des", des,
            "--itr", "1",
            "--use_gpu", "true",
            "--gpu", "0",
            "--checkpoints", str(output_dir / "checkpoints") + "/",
        ]
        params = count_params(
            repo,
            python,
            [
                "--seq_len", str(args.seq_len),
                "--pred_len", str(args.pred_len),
                "--d_model", str(args.d_model),
                "--n_heads", str(args.n_heads),
                "--e_layers", str(args.e_layers),
                "--d_ff", str(args.d_ff),
            ],
        )
        profile_args = [
            "--seq_len", str(args.seq_len),
            "--pred_len", str(args.pred_len),
            "--d_model", str(args.d_model),
            "--n_heads", str(args.n_heads),
            "--e_layers", str(args.e_layers),
            "--d_layers", str(args.d_layers),
            "--d_ff", str(args.d_ff),
        ]
        if "clean" in args.stresses:
            train_cmd = [python, "run.py", "--is_training", "1", *common]
            elapsed, _ = run_command(train_cmd, repo, env)
            sync_checkpoint_for_eval(repo, output_dir, base_setting)
            mse, mae = parse_metrics(repo, base_setting)
            checkpoint = output_dir / "checkpoints" / base_setting / "checkpoint.pth"
            latency = profile_latency(
                repo,
                python,
                checkpoint,
                profile_args,
                meta["enc_in"],
                args.latency_iters,
                args.latency_warmup,
                args.gpu,
            )
            copy_outputs(repo, base_setting, output_dir, dataset, "clean")
            clean_rows.append({
                "dataset": dataset,
                "stress": "clean",
                "model": "official_itransformer",
                "mse": mse,
                "mae": mae,
                "train_seconds": elapsed,
                "params": params,
                "enc_in": meta["enc_in"],
                **latency,
            })
        else:
            sync_checkpoint_for_eval(repo, output_dir, base_setting)

        for stress in args.stresses:
            if stress == "clean":
                continue
            stress_args = []
            for key, value in STRESSES[stress].items():
                stress_args.extend([key, value])
            eval_cmd = [python, "run.py", "--is_training", "0", *common, *stress_args]
            elapsed, _ = run_command(eval_cmd, repo, env)
            result_setting = stress_setting(base_setting, stress)
            mse, mae = parse_metrics(repo, result_setting)
            checkpoint = output_dir / "checkpoints" / base_setting / "checkpoint.pth"
            latency = profile_latency(
                repo,
                python,
                checkpoint,
                profile_args,
                meta["enc_in"],
                args.latency_iters,
                args.latency_warmup,
                args.gpu,
            )
            copy_outputs(repo, result_setting, output_dir, dataset, stress)
            stress_rows.append({
                "dataset": dataset,
                "stress": stress,
                "model": "official_itransformer",
                "mse": mse,
                "mae": mae,
                "eval_seconds": elapsed,
                "params": params,
                "enc_in": meta["enc_in"],
                **latency,
            })

    write_rows(clean_path, clean_rows)
    write_rows(stress_path, stress_rows)

    print(pd.DataFrame(clean_rows).to_string(index=False) if clean_rows else "No clean rows.")
    print(pd.DataFrame(stress_rows).to_string(index=False) if stress_rows else "No stress rows.")


if __name__ == "__main__":
    main()
