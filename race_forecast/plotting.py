from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_case_study(pred: np.ndarray, true: np.ndarray, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    idx = int(np.argmax(np.mean(np.abs(pred - true), axis=1)))
    x = np.arange(true.shape[1])
    plt.figure(figsize=(8, 3.2))
    plt.plot(x, true[idx], label="actual", linewidth=2)
    plt.plot(x, pred[idx], label="forecast", linewidth=2)
    plt.fill_between(x, np.minimum(true[idx], pred[idx]), np.maximum(true[idx], pred[idx]), alpha=0.18)
    plt.title(title)
    plt.xlabel("forecast horizon")
    plt.ylabel("normalized target load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()

