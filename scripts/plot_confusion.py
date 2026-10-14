"""Plot IMU / TP confusion matrices from saved metrics JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

from grip_classify import LABELS


def _setup_cjk_font() -> None:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def plot_cm(cm: np.ndarray, title: str, out_path: Path) -> None:
    cm = np.asarray(cm, dtype=np.int64)
    row_sum = cm.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1, row_sum)
    rates = cm / row_sum

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(rates, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row recall")

    ax.set_xticks(np.arange(len(LABELS)))
    ax.set_yticks(np.arange(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    thresh = 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if rates[i, j] >= thresh else "black"
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n{rates[i, j] * 100:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
            )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"saved {out_path}")


def main() -> None:
    _setup_cjk_font()
    artifacts = ROOT / "artifacts"
    for name, title in (
        ("imu_metrics.json", "IMU 150ms 三分类混淆矩阵"),
        ("tp_metrics.json", "TP 150ms 三分类混淆矩阵"),
    ):
        with open(artifacts / name, encoding="utf-8") as f:
            metrics = json.load(f)
        acc = metrics["accuracy"]
        stem = "imu" if name.startswith("imu") else "tp"
        plot_cm(
            np.array(metrics["confusion_matrix"]),
            f"{title}  (acc={acc:.3f})",
            artifacts / f"{stem}_confusion_matrix.png",
        )


if __name__ == "__main__":
    main()
