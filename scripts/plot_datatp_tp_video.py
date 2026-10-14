"""Render datatp TP grid streams to MP4 (video only, no still images)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grip_classify.data import (  # noqa: E402
    list_recording_sessions,
    load_tp_diff,
)

CELL_PX = 18
FPS = 30.0  # playback fps (source ~83 Hz @ 12ms)
SPEED = 1.0  # 1.0 = realtime wall clock
CMAP = colormaps["coolwarm"]


def _upsample(grid: np.ndarray, cell: int) -> np.ndarray:
    return np.repeat(np.repeat(grid, cell, axis=0), cell, axis=1)


def _to_rgb(grid: np.ndarray, vmax: float) -> np.ndarray:
    if vmax <= 0:
        vmax = 1.0
    norm = np.clip((grid / vmax + 1.0) * 0.5, 0.0, 1.0)
    rgba = CMAP(norm)
    return (rgba[..., :3] * 255.0).astype(np.uint8)


def _draw_text_bar(w: int, text: str, h: int = 36) -> np.ndarray:
    """Simple dark status bar via matplotlib text -> RGB array."""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(w / 100.0, h / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.text(0.01, 0.5, text, color="white", fontsize=11, va="center", ha="left", family="DejaVu Sans")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if buf.shape[1] != w or buf.shape[0] != h:
        # nearest resize without cv2
        ys = (np.linspace(0, buf.shape[0] - 1, h)).astype(np.int32)
        xs = (np.linspace(0, buf.shape[1] - 1, w)).astype(np.int32)
        buf = buf[ys][:, xs]
    return buf


def _mark_touch(rgb: np.ndarray, nx: float, ny: float, color=(0, 255, 80)) -> None:
    """Draw a small cross at normalized screen coords mapped onto TP grid image."""
    h, w = rgb.shape[:2]
    x = int(np.clip(nx, 0, 1) * (w - 1))
    y = int(np.clip(ny, 0, 1) * (h - 1))
    r = 6
    for dy in range(-r, r + 1):
        yy = y + dy
        if 0 <= yy < h and 0 <= x < w:
            rgb[yy, x] = color
        xx = x + dy
        if 0 <= y < h and 0 <= xx < w:
            rgb[y, xx] = color


def load_touch_markers(meta_path: Path) -> list[tuple[float, float, float, str]]:
    """Return (t_ms, nx, ny, phase) from metadata."""
    if not meta_path.is_file():
        return []
    with meta_path.open(encoding="utf-8") as f:
        meta = json.load(f)
    out = []
    for ev in meta.get("touchEvents") or []:
        ts = ev.get("timestampMs")
        if ts is None:
            continue
        nx = ev.get("normalizedX", ev.get("surfaceNormalizedX"))
        ny = ev.get("normalizedY", ev.get("surfaceNormalizedY"))
        if nx is None or ny is None:
            continue
        out.append((float(ts), float(nx), float(ny), str(ev.get("phase") or "")))
    return out


def render_session(
    label: str,
    sid: str,
    tp_path: Path,
    meta_path: Path,
    out_path: Path,
) -> None:
    tp = load_tp_diff(tp_path)
    # Real-time first-order diffs only (no cumsum reconstruction).
    grid = tp["grid_diff"]
    t_ms = tp["t_ms"]
    t0 = float(t_ms[0])
    poll = float(tp["poll_interval_ms"] or 12.0)

    # Stable color scale from robust abs percentile of per-frame diffs.
    vmax = float(np.percentile(np.abs(grid), 99.0))
    vmax = max(vmax, 1.0)

    touches = load_touch_markers(meta_path)
    tap_downs = [t for t, _, _, ph in touches if ph == "down"]

    # Source index schedule at playback FPS / SPEED.
    dur_s = (float(t_ms[-1]) - t0) / 1000.0
    n_out = max(1, int(round(dur_s / SPEED * FPS)))
    src_idx = np.clip(
        np.round(np.linspace(0, len(grid) - 1, n_out)).astype(np.int32),
        0,
        len(grid) - 1,
    )

    h = grid.shape[1] * CELL_PX
    w = grid.shape[2] * CELL_PX
    bar_h = 40
    frame_h = h + bar_h

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{w}x{frame_h}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    assert proc.stdin is not None
    last_bar_key = None
    bar = None
    try:
        for i, si in enumerate(src_idx):
            frame_grid = grid[si]
            rgb = _to_rgb(_upsample(frame_grid, CELL_PX), vmax)

            cur_t = float(t_ms[si])
            # latest touch within 120ms for marker flash
            for tt, nx, ny, ph in touches:
                if abs(tt - cur_t) <= 120.0:
                    color = (0, 255, 80) if ph == "down" else (255, 220, 40) if ph == "up" else (180, 180, 255)
                    _mark_touch(rgb, nx, ny, color)

            elapsed = (cur_t - t0) / 1000.0
            n_taps = sum(1 for td in tap_downs if td <= cur_t)
            bar_key = (int(elapsed * 10), n_taps, si)
            if bar is None or bar_key != last_bar_key:
                title = (
                    f"{label}/{sid}  realtime-diff  t={elapsed:6.2f}s  "
                    f"frame={si+1}/{len(grid)}  vmax={vmax:.0f}  "
                    f"taps={n_taps}/{len(tap_downs)}  poll={poll:.0f}ms"
                )
                bar = _draw_text_bar(w, title, h=bar_h)
                last_bar_key = bar_key

            frame = np.vstack([bar, rgb])
            proc.stdin.write(frame.tobytes())
            if (i + 1) % 200 == 0 or i == n_out - 1:
                print(f"  [{label}/{sid}] {i+1}/{n_out} frames", flush=True)
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg failed ({code}):\n{err[-1500:]}")
    print(f"[ok] {out_path}  ({dur_s:.1f}s @ {FPS}fps, {n_out} frames)", flush=True)


def main() -> None:
    data_root = ROOT / "data" / "datatp"
    out_dir = ROOT / "artifacts" / "datatp_tp_videos"
    folders = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    n = 0
    for folder in folders:
        sessions = list_recording_sessions(folder)
        if not sessions:
            print(f"[skip] {folder.name}: no sessions")
            continue
        for sid, _imu, tp_path, meta_path in sessions:
            name = f"{folder.name}_s{sid}.mp4"
            render_session(folder.name, sid, tp_path, meta_path, out_dir / name)
            n += 1
    print(f"done: {n} videos -> {out_dir}")


if __name__ == "__main__":
    main()
