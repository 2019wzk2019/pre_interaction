"""Online-style inference helpers for IMU / TP 150ms windows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from .data import load_imu, load_tp_diff, reconstruct_tp_absolute
from .features import (
    imu_window_features,
    resample_imu_window,
    tp_window_features,
)


def load_bundle(path: Path) -> dict:
    return joblib.load(path)


def _decode(bundle: dict, pred_ids: np.ndarray, probas: Optional[np.ndarray], meta: List[dict]) -> List[Dict]:
    labels = bundle["labels"]
    out = []
    for i, m in enumerate(meta):
        item = dict(m)
        pid = int(pred_ids[i])
        item["label"] = labels[pid]
        item["label_id"] = pid
        if probas is not None:
            item["proba"] = {labels[j]: float(probas[i, j]) for j in range(len(labels))}
        out.append(item)
    return out


def run_imu_file(
    model_path: Path,
    imu_csv: Path,
    stride_ms: float = 50.0,
) -> List[Dict]:
    bundle = load_bundle(model_path)
    imu = load_imu(imu_csv)
    window_ms = float(bundle["window_ms"])
    t_min = max(imu["acc"][0, 0], imu["gyro"][0, 0]) + window_ms
    t_max = min(imu["acc"][-1, 0], imu["gyro"][-1, 0])

    feats = []
    meta = []
    for t_end in np.arange(t_min, t_max + 1e-6, stride_ms):
        seq = resample_imu_window(imu, float(t_end), window_ms=window_ms)
        if seq is None:
            continue
        feats.append(imu_window_features(seq))
        meta.append({"t_end_ms": float(t_end)})
    if not feats:
        return []

    X = np.asarray(feats, dtype=np.float64)
    pipe = bundle["pipeline"]
    # Avoid nested joblib parallelism hang on Windows during batch predict
    rf = pipe.named_steps.get("rf")
    old_jobs = getattr(rf, "n_jobs", None)
    if rf is not None:
        rf.n_jobs = 1
    try:
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None
    finally:
        if rf is not None and old_jobs is not None:
            rf.n_jobs = old_jobs
    return _decode(bundle, pred, proba, meta)


def run_tp_file(
    model_path: Path,
    tp_jsonl: Path,
    window_ms: float = 150.0,
    stride_ms: float = 50.0,
) -> List[Dict]:
    bundle = load_bundle(model_path)
    raw = load_tp_diff(tp_jsonl)
    grid_abs, line_abs = reconstruct_tp_absolute(raw["grid_diff"], raw["line_diff"])
    tp = {
        "t_ms": raw["t_ms"],
        "grid_diff": raw["grid_diff"],
        "line_diff": raw["line_diff"],
        "grid_abs": grid_abs,
        "line_abs": line_abs,
    }
    dts = np.diff(tp["t_ms"])
    dt = float(np.median(dts)) if len(dts) else 14.0
    n_frames = max(3, int(round(window_ms / max(dt, 1.0))))
    stride_frames = max(1, int(round(stride_ms / max(dt, 1.0))))

    feats = []
    meta = []
    for idx_end in range(n_frames - 1, len(tp["t_ms"]), stride_frames):
        feat = tp_window_features(
            tp["grid_abs"],
            tp["line_abs"],
            tp["grid_diff"],
            tp["line_diff"],
            idx_end,
            n_frames,
        )
        if feat is None:
            continue
        feats.append(feat)
        meta.append({"t_end_ms": float(tp["t_ms"][idx_end]), "frame_index": idx_end})
    if not feats:
        return []

    X = np.asarray(feats, dtype=np.float64)
    pipe = bundle["pipeline"]
    rf = pipe.named_steps.get("rf")
    old_jobs = getattr(rf, "n_jobs", None)
    if rf is not None:
        rf.n_jobs = 1
    try:
        pred = pipe.predict(X)
        proba = pipe.predict_proba(X) if hasattr(pipe, "predict_proba") else None
    finally:
        if rf is not None and old_jobs is not None:
            rf.n_jobs = old_jobs
    return _decode(bundle, pred, proba, meta)


def predict_imu_at(bundle: dict, imu: dict, t_end_ms: float) -> Optional[Dict]:
    window_ms = float(bundle["window_ms"])
    seq = resample_imu_window(imu, t_end_ms, window_ms=window_ms)
    if seq is None:
        return None
    feat = imu_window_features(seq).reshape(1, -1)
    pipe = bundle["pipeline"]
    pred_id = int(pipe.predict(feat)[0])
    proba = pipe.predict_proba(feat)[0] if hasattr(pipe, "predict_proba") else None
    labels = bundle["labels"]
    out = {"label": labels[pred_id], "label_id": pred_id, "t_end_ms": t_end_ms}
    if proba is not None:
        out["proba"] = {labels[i]: float(p) for i, p in enumerate(proba)}
    return out
