"""Detect gyro-onset cuts and plot IMU with dashed=cut, solid=keep."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BASELINE_S = 0.12
K_NOISE = 5.0
MIN_DELTA = 0.05
HOLD_N = 12  # consecutive samples above thr (~24ms at 500Hz)


def load_imu_abs(path: Path) -> dict[str, np.ndarray] | None:
    """Return acc/gyro as (N,4) [wall_ms, x, y, z] sorted."""
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
                [float(row["wall_ms"]), float(row["x"]), float(row["y"]), float(row["z"])]
            )
    if not buckets["acc"] and not buckets["gyro"]:
        return None
    out = {}
    for sensor, rows in buckets.items():
        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        out[sensor] = arr[np.argsort(arr[:, 0], kind="mergesort")]
    return out


def load_tap_wall_ms(touch_path: Path) -> float | None:
    if not touch_path.is_file():
        return None
    with touch_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "type" not in reader.fieldnames:
            return None
        for row in reader:
            if (row.get("type") or "").strip() == "Down":
                return float(row["wall_ms"])
    return None


def detect_onset_s(gyro: np.ndarray, t0_ms: float) -> tuple[float, dict]:
    """gyro (N,4) absolute wall_ms. Return onset seconds relative to t0_ms."""
    t = (gyro[:, 0] - t0_ms) / 1000.0
    mag = np.linalg.norm(gyro[:, 1:], axis=1)
    early = mag[t <= BASELINE_S]
    if len(early) < 5:
        early = mag[: max(5, len(mag) // 10)]
    baseline = float(np.median(early))
    noise = float(np.std(early))
    thr = baseline + max(MIN_DELTA, K_NOISE * noise)

    onset_idx = None
    for i in range(len(mag)):
        j = min(i + HOLD_N, len(mag))
        if mag[i] > thr and float(np.mean(mag[i:j])) > thr * 0.85:
            onset_idx = i
            break
    if onset_idx is None:
        onset_s = 0.0
    else:
        onset_s = float(t[onset_idx])
    meta = {
        "baseline": baseline,
        "noise": noise,
        "thr": thr,
        "onset_s": onset_s,
        "onset_idx": int(onset_idx) if onset_idx is not None else None,
    }
    return onset_s, meta


def _plot_segmented(ax, t, y, cut_s, end_s, color, label):
    """Dashed for t < cut_s, solid for cut_s <= t <= end_s, faint after end."""
    keep = (t >= cut_s) & (t <= end_s)
    cut = t < cut_s
    after = t > end_s
    if np.any(cut):
        # include first keep point so line connects visually
        idx = np.where(cut)[0]
        if np.any(keep):
            first_keep = np.where(keep)[0][0]
            idx = np.concatenate([idx, [first_keep]])
        ax.plot(t[idx], y[idx], ls="--", lw=1.0, color=color, alpha=0.55, label=None)
    if np.any(keep):
        ax.plot(t[keep], y[keep], ls="-", lw=1.2, color=color, label=label)
    if np.any(after):
        idx = np.where(after)[0]
        if np.any(keep):
            last_keep = np.where(keep)[0][-1]
            idx = np.concatenate([[last_keep], idx])
        ax.plot(t[idx], y[idx], ls=":", lw=0.9, color=color, alpha=0.35, label=None)


def plot_cut(imu_path: Path, out_path: Path) -> dict | None:
    data = load_imu_abs(imu_path)
    if not data or "gyro" not in data:
        print(f"[skip] {imu_path.name}")
        return None

    # In this dump, IMU stream is *_tp.csv; touch Down is sibling *_imu.csv
    stem = imu_path.name
    if stem.endswith("_tp.csv"):
        touch_path = imu_path.with_name(stem.replace("_tp.csv", "_imu.csv"))
    else:
        touch_path = imu_path.with_name(stem.replace("_imu.csv", "_touch.csv"))

    t0 = float(
        min(arr[0, 0] for arr in data.values() if len(arr))
    )
    t_end_abs = float(max(arr[-1, 0] for arr in data.values() if len(arr)))
    rec_end_s = (t_end_abs - t0) / 1000.0
    tap_abs = load_tap_wall_ms(touch_path)
    onset_s, meta = detect_onset_s(data["gyro"], t0)
    # Tap must fall inside this IMU clip (+50ms slack). Otherwise clocks/pairing are off.
    tap_ok = (
        tap_abs is not None
        and (t0 - 20.0) <= tap_abs <= (t_end_abs + 50.0)
    )
    if tap_ok:
        end_s = (tap_abs - t0) / 1000.0
    else:
        end_s = rec_end_s
    if end_s < onset_s:
        end_s = rec_end_s

    is_others = "OTHERS" in imu_path.name.upper()
    colors = {"x": "#1f77b4", "y": "#ff7f0e", "z": "#2ca02c"}

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 8.2), sharex=True)
    # acc / gyro
    for ax, sensor, ylabel in (
        (axes[0], "acc", "Accelerometer (m/s$^2$)"),
        (axes[1], "gyro", "Gyroscope (rad/s)"),
    ):
        arr = data.get(sensor)
        if arr is None:
            continue
        t = (arr[:, 0] - t0) / 1000.0
        for i, name in enumerate(("x", "y", "z")):
            _plot_segmented(ax, t, arr[:, i + 1], onset_s, end_s, colors[name], name)
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axvline(onset_s, color="crimson", lw=1.3, label="onset")
        ax.axvline(end_s, color="purple", lw=1.3, ls="-.", label="tap/end")
        # only put vline labels once
        handles, labels = ax.get_legend_handles_labels()
        # rebuild legend with xyz + markers
        ax.legend(loc="upper right", ncol=5, fontsize=8)

    # gyro magnitude panel for clarity
    g = data["gyro"]
    tg = (g[:, 0] - t0) / 1000.0
    mag = np.linalg.norm(g[:, 1:], axis=1)
    axm = axes[2]
    _plot_segmented(axm, tg, mag, onset_s, end_s, "#444444", "|gyro|")
    axm.axhline(meta["thr"], color="crimson", lw=1.0, ls=":", alpha=0.8, label=f"thr={meta['thr']:.3f}")
    axm.axvline(onset_s, color="crimson", lw=1.3)
    axm.axvline(end_s, color="purple", lw=1.3, ls="-.")
    axm.set_ylabel("|gyro| (rad/s)")
    axm.set_xlabel("relative time (s)")
    axm.legend(loc="upper right", fontsize=8)
    axm.grid(True, alpha=0.3)

    keep_dur = max(0.0, end_s - onset_s)
    tag = "OTHERS (onset shown, usually keep all)" if is_others else "L2R/R2L"
    title = (
        f"{imu_path.name}\n"
        f"{tag}  cut=[0,{onset_s:.3f})s dashed  |  keep=[{onset_s:.3f},{end_s:.3f}]s solid  "
        f"({keep_dur*1000:.0f} ms)  tap={'yes' if tap_ok else 'no/mismatch→rec end'}"
    )
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)

    rec = {
        "file": imu_path.name,
        "onset_s": onset_s,
        "end_s": end_s,
        "keep_ms": keep_dur * 1000.0,
        "tap_found": bool(tap_ok),
        "tap_raw_found": tap_abs is not None,
        "is_others": is_others,
        **{k: meta[k] for k in ("baseline", "noise", "thr")},
    }
    print(
        f"[ok] {out_path.name}  onset={onset_s:.3f}s  end={end_s:.3f}s  keep={keep_dur*1000:.0f}ms"
    )
    return rec


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "imu_old"
    out_dir = Path(__file__).resolve().parents[1] / "data" / "plots" / "imu_cut"
    # Real IMU streams are currently named *_tp.csv in this dump
    files = sorted(root.glob("*_tp.csv"))
    records = []
    for p in files:
        rec = plot_cut(p, out_dir / f"{p.stem}_cut.png")
        if rec:
            records.append(rec)
    summary = out_dir / "cut_summary.json"
    summary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(records)} figures -> {out_dir}")
    print(f"summary -> {summary}")


if __name__ == "__main__":
    main()
