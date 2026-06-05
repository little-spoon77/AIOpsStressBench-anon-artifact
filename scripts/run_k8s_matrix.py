import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(" ".join(cmd), flush=True)
    return subprocess.run(cmd, check=check, text=True)


def copy_outputs(out_dir: Path, prefix: str) -> None:
    mapping = {
        "sandbox_episode_summary_full.csv": f"{prefix}_summary.csv",
        "sandbox_policy_trace_full.csv": f"{prefix}_trace.csv",
        "sandbox_decision_gate.md": f"{prefix}_decision.md",
    }
    for src_name, dst_name in mapping.items():
        src = out_dir / src_name
        if src.exists():
            shutil.copy2(src, out_dir / dst_name)


def set_burn_scale(workload: Path, value: int) -> None:
    text = workload.read_text(encoding="utf-8")
    import re

    text = re.sub(r'(name: BURN_SCALE\s+value: ")[0-9]+(")', rf"\g<1>{value}\2", text)
    workload.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kubeconfig", required=True)
    parser.add_argument("--namespace", default="aiops-stressbench-sandbox")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--workload", default="sandbox/aiops/k8s/workload.yaml")
    parser.add_argument("--probe", default="sandbox/aiops/run_sandbox_probe.py")
    parser.add_argument("--out-dir", default="outputs/strong_probe/k8s_sandbox")
    parser.add_argument("--burn-scale", type=int, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--trace-npz", required=True)
    parser.add_argument("--trace-scale-min", type=float, required=True)
    parser.add_argument("--trace-scale-max", type=float, required=True)
    parser.add_argument("--target-per-replica", type=float, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--stresses", required=True)
    parser.add_argument("--policies", default="reactive,dlinear,patchtst,stressroute")
    parser.add_argument("--scale-delay-steps", type=int, default=3)
    parser.add_argument("--forecast-horizon", type=int, default=4)
    parser.add_argument("--metrics-every", type=int, default=10)
    parser.add_argument("--request-timeout-s", type=float, default=2.0)
    parser.add_argument("--max-requests-per-step", type=int, default=1)
    args = parser.parse_args()

    root = Path(args.repo_root)
    workload = root / args.workload
    probe = root / args.probe
    out_dir = root / args.out_dir
    set_burn_scale(workload, args.burn_scale)

    kubectl_base = ["kubectl", "--kubeconfig", args.kubeconfig, "-n", args.namespace]
    run([*kubectl_base, "apply", "-f", str(workload)])
    run([*kubectl_base, "rollout", "status", "deployment/aiops-workload", "--timeout=120s"])

    try:
        run(
            [
                sys.executable,
                str(probe),
                "--kubeconfig",
                args.kubeconfig,
                "--namespace",
                args.namespace,
                "--port",
                str(args.port),
                "--steps",
                str(args.steps),
                "--interval-s",
                "0.1",
                "--seeds",
                str(args.seeds),
                "--policies",
                args.policies,
                "--stresses",
                args.stresses,
                "--trace-npz",
                args.trace_npz,
                "--trace-scale-min",
                str(args.trace_scale_min),
                "--trace-scale-max",
                str(args.trace_scale_max),
                "--target-per-replica",
                str(args.target_per_replica),
                "--max-replicas",
                "8",
                "--scale-delay-steps",
                str(args.scale_delay_steps),
                "--forecast-horizon",
                str(args.forecast_horizon),
                "--metrics-every",
                str(args.metrics_every),
                "--request-timeout-s",
                str(args.request_timeout_s),
                "--max-requests-per-step",
                str(args.max_requests_per_step),
            ]
        )
        copy_outputs(out_dir, args.prefix)
    finally:
        run([*kubectl_base, "scale", "deployment/aiops-workload", "--replicas=0"], check=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
