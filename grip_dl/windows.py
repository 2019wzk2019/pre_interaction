"""TP frame windows for the DL model (raw diffs, no RF stats).

Default shipped model uses one raw frame. window_ms>0 keeps the older resampled windows.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from grip_classify.data import window_overlaps_excludes
from grip_classify.features import _long_edge
from grip_dl import N_FRAMES, STRIDE_MS, WINDOW_MS


def resample_time(seq: np.ndarray, n_out: int) -> np.ndarray:
    """seq (T, ...) -> (n_out, ...) linear interp along time (vectorized)."""
    t_in = int(seq.shape[0])
    rest = seq.shape[1:]
    if t_in == n_out:
        return np.ascontiguousarray(seq, dtype=np.float32)
    flat = np.ascontiguousarray(seq.reshape(t_in, -1), dtype=np.float32)
    if t_in < 2:
        return np.repeat(flat[:1], n_out, axis=0).reshape((n_out,) + rest)
    idx = np.linspace(0.0, t_in - 1.0, n_out, dtype=np.float32)
    i0 = np.floor(idx).astype(np.int32)
    i1 = np.minimum(i0 + 1, t_in - 1)
    w = (idx - i0).astype(np.float32)[:, None]
    out = (1.0 - w) * flat[i0] + w * flat[i1]
    return out.reshape((n_out,) + rest)


def extract_tp_sequences(
    recordings: dict,
    window_ms: float = WINDOW_MS,
    stride_ms: float = STRIDE_MS,
    n_frames: int = N_FRAMES,
    edge_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      X: (N, T, 34, 21) float32 raw first-order diffs on the full TP grid
      y: (N,) int64 class ids
    """
    xs: List[np.ndarray] = []
    ys: List[int] = []
    single_frame = n_frames == 1 and window_ms <= 0
    for rec in recordings.values():
        tp = rec["tp"]
        label_id = int(rec["label_id"])
        t = tp["t_ms"]
        grid_diff = tp["grid_diff"]
        excludes = rec.get("exclude_intervals") or []
        if single_frame:
            n_skip = 0
            for i, ts in enumerate(t):
                ts = float(ts)
                if window_overlaps_excludes(ts, ts, excludes):
                    n_skip += 1
                    continue
                xs.append(np.ascontiguousarray(grid_diff[i : i + 1], dtype=np.float32))
                ys.append(label_id)
            if n_skip:
                print(f"[tp exclude] {rec.get('folder', '?')}: skipped {n_skip} frames")
            continue
        dts = np.diff(t)
        dt = float(np.median(dts)) if len(dts) else float(tp.get("poll_interval_ms", 14))
        win_frames = max(3, int(round(window_ms / max(dt, 1.0))))
        stride_frames = max(1, int(round(stride_ms / max(dt, 1.0))))
        n_skip = 0
        for idx_end in range(win_frames - 1, len(t), stride_frames):
            i0 = idx_end - win_frames + 1
            t0 = float(t[i0])
            t1 = float(t[idx_end])
            if window_overlaps_excludes(t0, t1, rec.get("exclude_intervals") or []):
                n_skip += 1
                continue
            gd = grid_diff[i0 : idx_end + 1]
            if edge_only:
                gd = _long_edge(gd)
            xs.append(resample_time(gd, n_frames))
            ys.append(label_id)
        if n_skip:
            print(f"[tp exclude] {rec.get('folder', '?')}: skipped {n_skip} windows")
    if not xs:
        h, w = (34, 6) if edge_only else (34, 21)
        return np.zeros((0, n_frames, h, w), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)
