"""Plot each IMU recording in data/imu_old as one figure. X = seconds from first sample."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_imu_csv(path: Path) -> dict[str, np.ndarray] | None:
    """kind=acc/gyro rows -> {sensor: (N,4) [t_s_rel, x, y, z]} using wall_ms, t relative to min."""
    buckets: dict[str, list[list[float]]] = {"acc": [], "gyro": []}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "kind" not in reader.fieldnames:
            return None
        for row in reader:
            kind = (row.get("kind") or "").strip()
            if kind not in buckets:
                continue
            buckets[kind].append(
                [
                    float(row["wall_ms"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                ]
            )
    if not buckets["acc"] and not buckets["gyro"]:
        return None
    t0 = min(r[0] for rows in buckets.values() for r in rows)
    out = {}
    for sensor, rows in buckets.items():
        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
        arr[:, 0] = (arr[:, 0] - t0) / 1000.0
        out[sensor] = arr
    return out


def plot_one(path: Path, out_path: Path) -> bool:
    data = load_imu_csv(path)
    if not data:
        print(f"[skip] not IMU stream: {path.name}")
        return False
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    specs = [
        ("acc", axes[0], "Accelerometer (m/s$^2$)"),
        ("gyro", axes[1], "Gyroscope (rad/s)"),
    ]
    t_max = 0.0
    for sensor, ax, ylabel in specs:
        arr = data.get(sensor)
        if arr is None or len(arr) == 0:
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ylabel} — no samples")
            continue
        t = arr[:, 0]
        t_max = max(t_max, float(t[-1]) if len(t) else 0.0)
        for i, name in enumerate(("x", "y", "z")):
            ax.plot(t, arr[:, i + 1], lw=0.9, label=name)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("relative time (s)")
    if t_max > 0:
        axes[-1].set_xlim(0, t_max)
    fig.suptitle(path.name, fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[ok] {out_path.name}  t=0..{t_max:.2f}s")
    return True


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "imu_old"
    out_dir = Path(__file__).resolve().parents[1] / "data" / "plots" / "imu"
    files = sorted(root.glob("*.csv"))
    n = 0
    for p in files:
        if p.name == "meta.csv":
            continue
        if plot_one(p, out_dir / f"{p.stem}.png"):
            n += 1
    print(f"wrote {n} figures -> {out_dir}")


if __name__ == "__main__":
    main()
