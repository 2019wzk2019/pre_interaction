"""150ms windowing and feature extraction for IMU / TP streams."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import STRIDE_MS, TP_EDGE_LAYERS, TP_GRID_COLS, WINDOW_MS
from .data import window_overlaps_excludes


def _interp_channel(t_src: np.ndarray, y_src: np.ndarray, t_query: np.ndarray) -> np.ndarray:
    if len(t_src) < 2:
        return np.full_like(t_query, y_src[0] if len(y_src) else 0.0, dtype=np.float64)
    return np.interp(t_query, t_src, y_src)


def resample_imu_window(
    imu: dict,
    t_end_ms: float,
    window_ms: float = WINDOW_MS,
    rate_hz: float = 100.0,
) -> Optional[np.ndarray]:
    """
    Resample acc+gyro into a fixed (T, 6) array ending at t_end_ms.
    Channels: ax, ay, az, gx, gy, gz.
    """
    t0 = t_end_ms - window_ms
    n = max(2, int(round(window_ms / 1000.0 * rate_hz)))
    t_query = np.linspace(t0, t_end_ms, n, dtype=np.float64)

    acc = imu["acc"]
    gyro = imu["gyro"]
    # need coverage
    if acc[0, 0] > t0 or acc[-1, 0] < t_end_ms:
        return None
    if gyro[0, 0] > t0 or gyro[-1, 0] < t_end_ms:
        return None

    cols = []
    for src in (acc, gyro):
        for c in range(1, 4):
            cols.append(_interp_channel(src[:, 0], src[:, c], t_query))
    return np.stack(cols, axis=1)  # (T, 6)


def imu_window_features(seq: np.ndarray) -> np.ndarray:
    """Handcrafted stats over (T, 6) IMU window."""
    feats: List[float] = []
    # raw end / start / delta
    feats.extend(seq[-1].tolist())
    feats.extend(seq[0].tolist())
    feats.extend((seq[-1] - seq[0]).tolist())
    # mean / std / min / max / range / energy / abs-mean
    for c in range(seq.shape[1]):
        x = seq[:, c]
        feats.extend(
            [
                float(np.mean(x)),
                float(np.std(x)),
                float(np.min(x)),
                float(np.max(x)),
                float(np.max(x) - np.min(x)),
                float(np.mean(x * x)),
                float(np.mean(np.abs(x))),
            ]
        )
    # magnitude stats
    acc = seq[:, :3]
    gyro = seq[:, 3:]
    acc_mag = np.linalg.norm(acc, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)
    for mag in (acc_mag, gyro_mag):
        feats.extend(
            [
                float(np.mean(mag)),
                float(np.std(mag)),
                float(np.max(mag)),
                float(np.mean(np.abs(np.diff(mag)))) if len(mag) > 1 else 0.0,
            ]
        )
    # gravity / tilt cues (help separate static 左 vs 右)
    gvec = np.mean(acc, axis=0)
    gnorm = float(np.linalg.norm(gvec) + 1e-8)
    ghat = gvec / gnorm
    feats.extend(ghat.tolist())
    feats.append(float(np.arctan2(ghat[1], ghat[0])))
    feats.append(float(np.arctan2(ghat[2], np.linalg.norm(ghat[:2]))))
    # gyro mean direction + jerk proxies
    feats.extend(np.mean(gyro, axis=0).tolist())
    if len(seq) > 2:
        jerk = np.diff(acc, axis=0)
        feats.extend(
            [
                float(np.mean(np.linalg.norm(jerk, axis=1))),
                float(np.std(np.linalg.norm(jerk, axis=1))),
            ]
        )
    else:
        feats.extend([0.0, 0.0])
    return np.asarray(feats, dtype=np.float64)


def extract_imu_windows(
    recordings: dict,
    window_ms: float = WINDOW_MS,
    stride_ms: float = STRIDE_MS,
    rate_hz: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """All valid IMU windows (after exclude intervals)."""
    X, y = [], []
    for rec in recordings.values():
        imu = rec["imu"]
        label_id = rec["label_id"]
        excludes = rec.get("exclude_intervals") or []
        t_min = max(imu["acc"][0, 0], imu["gyro"][0, 0]) + window_ms
        t_max = min(imu["acc"][-1, 0], imu["gyro"][-1, 0])
        ends = np.arange(t_min, t_max + 1e-6, stride_ms)
        n_skip = 0
        for t_end in ends:
            t_end = float(t_end)
            if window_overlaps_excludes(t_end - window_ms, t_end, excludes):
                n_skip += 1
                continue
            seq = resample_imu_window(imu, t_end, window_ms, rate_hz)
            if seq is None:
                continue
            X.append(imu_window_features(seq))
            y.append(label_id)
        if n_skip:
            print(f"[imu exclude] {rec.get('folder', '?')}: skipped {n_skip} windows")
    if not X:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def make_imu_dataset(
    recordings: dict,
    window_ms: float = WINDOW_MS,
    stride_ms: float = STRIDE_MS,
    rate_hz: float = 100.0,
    train_frac: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Temporal split per recording: first train_frac -> train, rest -> test."""
    X_train, y_train, X_test, y_test = [], [], [], []
    for rec in recordings.values():
        imu = rec["imu"]
        label_id = rec["label_id"]
        excludes = rec.get("exclude_intervals") or []
        t_min = max(imu["acc"][0, 0], imu["gyro"][0, 0]) + window_ms
        t_max = min(imu["acc"][-1, 0], imu["gyro"][-1, 0])
        ends = np.arange(t_min, t_max + 1e-6, stride_ms)
        split_t = t_min + (t_max - t_min) * train_frac
        n_skip = 0
        for t_end in ends:
            t_end = float(t_end)
            if window_overlaps_excludes(t_end - window_ms, t_end, excludes):
                n_skip += 1
                continue
            seq = resample_imu_window(imu, t_end, window_ms, rate_hz)
            if seq is None:
                continue
            feat = imu_window_features(seq)
            if t_end <= split_t:
                X_train.append(feat)
                y_train.append(label_id)
            else:
                X_test.append(feat)
                y_test.append(label_id)
        if n_skip:
            print(f"[imu exclude] {rec.get('folder', '?')}: skipped {n_skip} windows")
    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.int64),
    )


def _pool_grid(grid: np.ndarray, pool: int = 2) -> np.ndarray:
    """Average-pool HxW grid for compact features."""
    h, w = grid.shape
    nh, nw = h // pool, w // pool
    g = grid[: nh * pool, : nw * pool].reshape(nh, pool, nw, pool).mean(axis=(1, 3))
    return g


def _long_edge(grid: np.ndarray, layers: int = TP_EDGE_LAYERS) -> np.ndarray:
    """Keep left/right long-edge columns only. grid (H, W) -> (H, 2*layers)."""
    w = grid.shape[-1]
    if w != TP_GRID_COLS:
        raise ValueError(f"expected {TP_GRID_COLS} cols, got {w}")
    return np.concatenate([grid[..., :layers], grid[..., -layers:]], axis=-1)


def _robust_unit(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    """Clip to [p_lo, p_hi] then scale to [0, 1]. Same idea as the GIF colormap."""
    p_lo = float(np.percentile(x, lo))
    p_hi = float(np.percentile(x, hi))
    if p_hi <= p_lo:
        p_hi = p_lo + 1.0
    return np.clip((x - p_lo) / (p_hi - p_lo), 0.0, 1.0)


def tp_window_features(
    grid_abs: np.ndarray,
    line_abs: np.ndarray,
    grid_diff: np.ndarray,
    line_diff: np.ndarray,
    idx_end: int,
    n_frames: int,
    robust_p2_p98: bool = True,
) -> Optional[np.ndarray]:
    """
    Features for TP window ending at frame idx_end (inclusive), length n_frames.

    Only the portrait long edges (left/right outermost TP_EDGE_LAYERS columns)
    are used. Inner cells and LineData are dropped to cut grip-noise in the
    center of the panel.

    If robust_p2_p98: integrated levels and diffs are 2–98 percentile clipped
    and scaled to [0, 1] inside the 150ms window.
    """
    i1 = idx_end + 1
    i0 = i1 - n_frames
    if i0 < 0:
        return None
    g_raw = _long_edge(grid_abs[i0:i1])
    gd = _long_edge(grid_diff[i0:i1])
    if robust_p2_p98:
        g_raw = _robust_unit(g_raw)
        gd = _robust_unit(gd)

    g = g_raw - g_raw.mean(axis=0, keepdims=True)

    feats: List[float] = []
    feats.extend(g[-1].ravel().tolist())
    mean_g = g_raw.mean(axis=0)
    feats.extend((mean_g - mean_g.mean()).ravel().tolist())
    feats.extend(g_raw.std(axis=0).ravel().tolist())
    feats.extend(gd[-1].ravel().tolist())
    feats.append(float(np.mean(np.abs(gd))))
    feats.extend((g_raw[-1] - g_raw[0]).ravel().tolist())
    return np.asarray(feats, dtype=np.float64)


def extract_tp_windows(
    recordings: dict,
    window_ms: float = WINDOW_MS,
    stride_ms: float = STRIDE_MS,
    robust_p2_p98: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """All valid TP windows (after exclude intervals)."""
    X, y = [], []
    for rec in recordings.values():
        tp = rec["tp"]
        label_id = rec["label_id"]
        t = tp["t_ms"]
        dts = np.diff(t)
        dt = float(np.median(dts)) if len(dts) else float(tp.get("poll_interval_ms", 14))
        n_frames = max(3, int(round(window_ms / max(dt, 1.0))))
        stride_frames = max(1, int(round(stride_ms / max(dt, 1.0))))
        ends = list(range(n_frames - 1, len(t), stride_frames))
        n_skip = 0
        for idx_end in ends:
            t0 = float(t[idx_end - n_frames + 1])
            t1 = float(t[idx_end])
            if window_overlaps_excludes(t0, t1, rec.get("exclude_intervals") or []):
                n_skip += 1
                continue
            feat = tp_window_features(
                tp["grid_abs"],
                tp["line_abs"],
                tp["grid_diff"],
                tp["line_diff"],
                idx_end,
                n_frames,
                robust_p2_p98=robust_p2_p98,
            )
            if feat is None:
                continue
            X.append(feat)
            y.append(label_id)
        if n_skip:
            print(f"[tp exclude] {rec.get('folder', '?')}: skipped {n_skip} windows")
    if not X:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64)


def make_tp_dataset(
    recordings: dict,
    window_ms: float = WINDOW_MS,
    stride_ms: float = STRIDE_MS,
    train_frac: float = 0.8,
    robust_p2_p98: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train, X_test, y_test = [], [], [], []
    for rec in recordings.values():
        tp = rec["tp"]
        label_id = rec["label_id"]
        t = tp["t_ms"]
        dts = np.diff(t)
        dt = float(np.median(dts)) if len(dts) else float(tp.get("poll_interval_ms", 14))
        n_frames = max(3, int(round(window_ms / max(dt, 1.0))))
        stride_frames = max(1, int(round(stride_ms / max(dt, 1.0))))

        ends = list(range(n_frames - 1, len(t), stride_frames))
        split_i = int(len(ends) * train_frac)
        n_skip = 0
        for k, idx_end in enumerate(ends):
            t0 = float(t[idx_end - n_frames + 1])
            t1 = float(t[idx_end])
            if window_overlaps_excludes(t0, t1, rec.get("exclude_intervals") or []):
                n_skip += 1
                continue
            feat = tp_window_features(
                tp["grid_abs"],
                tp["line_abs"],
                tp["grid_diff"],
                tp["line_diff"],
                idx_end,
                n_frames,
                robust_p2_p98=robust_p2_p98,
            )
            if feat is None:
                continue
            if k < split_i:
                X_train.append(feat)
                y_train.append(label_id)
            else:
                X_test.append(feat)
                y_test.append(label_id)
        if n_skip:
            print(f"[tp exclude] {rec.get('folder', '?')}: skipped {n_skip} windows")
    return (
        np.asarray(X_train, dtype=np.float64),
        np.asarray(y_train, dtype=np.int64),
        np.asarray(X_test, dtype=np.float64),
        np.asarray(y_test, dtype=np.int64),
    )
