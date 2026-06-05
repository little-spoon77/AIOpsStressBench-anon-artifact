#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def check_env(name: str, python_path: Path) -> None:
    if not python_path.exists():
        return
    code = (
        "import sys\n"
        "try:\n"
        " import torch\n"
        " print('PY=%d.%d TORCH=%s CUDA=%s CUDA_VER=%s' % "
        "(sys.version_info.major, sys.version_info.minor, torch.__version__, "
        "torch.cuda.is_available(), getattr(torch.version, 'cuda', None)))\n"
        "except Exception as ex:\n"
        " print('PY=%d.%d NO_TORCH %s: %s' % "
        "(sys.version_info.major, sys.version_info.minor, type(ex).__name__, ex))\n"
    )
    proc = subprocess.run(
        [str(python_path), "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    out = proc.stdout.strip()
    if "TORCH=" in out:
        print(f"{name} :: {out}")


def main() -> None:
    home = Path(os.path.expanduser("~"))
    check_env("base", home / "anaconda3" / "bin" / "python")
    env_root = home / "anaconda3" / "envs"
    for env_dir in sorted(env_root.glob("*")):
        check_env(env_dir.name, env_dir / "bin" / "python")


if __name__ == "__main__":
    main()
