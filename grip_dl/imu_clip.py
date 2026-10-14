"""Load IMU CSVs, onset-cut L2R/R2L, leave OTHERS uncut, slice windows.

Two protocols:
- old: whole post-onset clip is the filename swipe label (morning l2r_* files).
- twophase: front half is swipe (onset-cut), back half is others (test_* files).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LABELS = ["左to右", "右to左", "others"]
LABEL_TO_ID = {n: i for i, n in enumerate(LABELS)}
_TAG_RE = re.compile(r"^(?:l2r|test)_(L2R|R2L|OTHERS)_", re.IGNORECASE)
_TAG_TO_LABEL = {"L2R": "左to右", "R2L": "右to左", "OTHERS": "others"}

BASELINE_S = 0.12
K_NOISE = 5.0
MIN_DELTA = 0.05
HOLD_N = 12
RATE_HZ = 100.0


@dataclass(frozen=True)
class ImuRec:
    path: Path
    protocol: str  # "old" | "twophase"

    @property
    def name(self) -> str:
        return self.path.name


def label_from_name(path: Path) -> Optional[str]:
    m = _TAG_RE.match(path.name)
    if not m:
        return None
    return _TAG_TO_LABEL[m.group(1).upper()]


def is_twophase_name(path: Path) -> bool:
    return path.name.lower().startswith("test_")


def load_imu_abs(path: Path) -> Optional[Dict[str, np.ndarray]]:
    buckets: Dict[str, List[List[float]]] = {"acc": [], "gyro": []}
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
    if not buckets["acc"] or not buckets["gyro"]:
        return None
    out = {}
    for sensor, rows in buckets.items():
        arr = np.asarray(rows, dtype=np.float64)
        out[sensor] = arr[np.argsort(arr[:, 0], kind="mergesort")]
    return out


def detect_onset_s(gyro: np.ndarray, t0_ms: float) -> float:
    t = (gyro[:, 0] - t0_ms) / 1000.0
    mag = np.linalg.norm(gyro[:, 1:], axis=1)
    early = mag[t <= BASELINE_S]
    if len(early) < 5:
        early = mag[: max(5, len(mag) // 10)]
    baseline = float(np.median(early))
    noise = float(np.std(early))
    thr = baseline + max(MIN_DELTA, K_NOISE * noise)
    for i in range(len(mag)):
        j = min(i + HOLD_N, len(mag))
        if mag[i] > thr and float(np.mean(mag[i:j])) > thr * 0.85:
            return float(t[i])
    return 0.0


def clip_range_ms(path: Path, imu: Dict[str, np.ndarray]) -> Tuple[float, float]:
    """Return [t0_ms, t1_ms] to keep. OTHERS: full file; reach: onset→end."""
    clips = labeled_clips(path, imu, protocol="old")
    if not clips:
        acc, gyro = imu["acc"], imu["gyro"]
        return float(min(acc[0, 0], gyro[0, 0])), float(max(acc[-1, 0], gyro[-1, 0]))
    return clips[0][0], clips[0][1]


def labeled_clips(
    path: Path,
    imu: Dict[str, np.ndarray],
    protocol: str = "old",
) -> List[Tuple[float, float, str]]:
    """(t0_ms, t1_ms, label_name) segments for one recording."""
    acc, gyro = imu["acc"], imu["gyro"]
    t_start = float(min(acc[0, 0], gyro[0, 0]))
    t_end = float(max(acc[-1, 0], gyro[-1, 0]))
    label = label_from_name(path)
    if label is None:
        return []
    if label == "others":
        return [(t_start, t_end, "others")]
    onset_ms = t_start + detect_onset_s(gyro, t_start) * 1000.0
    if protocol == "twophase":
        t_mid = 0.5 * (t_start + t_end)
        clips: List[Tuple[float, float, str]] = []
        swipe_end = min(t_mid, t_end)
        if swipe_end - onset_ms > 20.0:
            clips.append((onset_ms, swipe_end, label))
        if t_end - t_mid > 20.0:
            clips.append((t_mid, t_end, "others"))
        return clips
    if t_end - onset_ms <= 20.0:
        return []
    return [(onset_ms, t_end, label)]


def resample_clip(
    imu: Dict[str, np.ndarray],
    t0_ms: float,
    t1_ms: float,
    rate_hz: float = RATE_HZ,
) -> Optional[np.ndarray]:
    """(T, 6) acc xyz + gyro xyz on a uniform grid covering [t0, t1]."""
    if t1_ms <= t0_ms + 20.0:
        return None
    n = max(2, int(round((t1_ms - t0_ms) / 1000.0 * rate_hz)))
    tq = np.linspace(t0_ms, t1_ms, n, dtype=np.float64)
    cols = []
    for key in ("acc", "gyro"):
        src = imu[key]
        if src[0, 0] > t0_ms + 5 or src[-1, 0] < t1_ms - 5:
            # allow slight coverage miss
            if src[0, 0] > t1_ms or src[-1, 0] < t0_ms:
                return None
        for c in range(1, 4):
            cols.append(np.interp(tq, src[:, 0], src[:, c]))
    return np.stack(cols, axis=1).astype(np.float32)


def slice_windows(seq: np.ndarray, window_ms: float, stride_ms: float, rate_hz: float = RATE_HZ) -> np.ndarray:
    """seq (T,6) -> (N, Tw, 6)."""
    win = max(2, int(round(window_ms / 1000.0 * rate_hz)))
    stride = max(1, int(round(stride_ms / 1000.0 * rate_hz)))
    if seq.shape[0] < win:
        return np.zeros((0, win, 6), dtype=np.float32)
    xs = [seq[i : i + win] for i in range(0, seq.shape[0] - win + 1, stride)]
    return np.stack(xs, axis=0) if xs else np.zeros((0, win, 6), dtype=np.float32)


def list_imu_files(data_root: Path) -> List[Path]:
    files = []
    for p in sorted(data_root.glob("*.csv")):
        if "_tp" not in p.name:
            continue
        if label_from_name(p) is None:
            continue
        files.append(p)
    return files


def rec_key(path: Path) -> str:
    """Collapse browser-download suffixes like ' (1).csv'."""
    return re.sub(r" \(\d+\)\.csv$", ".csv", path.name, flags=re.IGNORECASE).lower()


def collect_imu_recs(project_root: Path) -> List[ImuRec]:
    """Old protocol: data/imu_old l2r_*. Two-phase: data/imu_new test_*."""
    recs: List[ImuRec] = []
    seen = set()
    for folder in (
        project_root / "data" / "imu_old",
        project_root / "data" / "imu_new",
        project_root / "data_old",
        project_root / "data_new",
    ):
        if not folder.exists():
            continue
        for p in list_imu_files(folder):
            key = rec_key(p)
            if key in seen:
                continue
            seen.add(key)
            proto = "twophase" if is_twophase_name(p) else "old"
            recs.append(ImuRec(p, proto))
    return recs


def split_files(files: List[Path], seed: int = 42) -> Tuple[List[Path], List[Path]]:
    recs = [ImuRec(p, "twophase" if is_twophase_name(p) else "old") for p in files]
    tr, te = split_recs(recs, seed=seed)
    return [r.path for r in tr], [r.path for r in te]


def split_recs(recs: List[ImuRec], seed: int = 42) -> Tuple[List[ImuRec], List[ImuRec]]:
    """Hold out swipe files per protocol; dedicated OTHERS stay in both (time-split)."""
    rng = np.random.default_rng(seed)
    train: List[ImuRec] = []
    test: List[ImuRec] = []
    for proto in ("old", "twophase"):
        by = {"左to右": [], "右to左": [], "others": []}
        for r in recs:
            if r.protocol != proto:
                continue
            by[label_from_name(r.path)].append(r)
        for lab in ("左to右", "右to左"):
            grp = list(by[lab])
            rng.shuffle(grp)
            n_te = 3 if len(grp) >= 6 else max(1, min(3, len(grp) // 3 + (1 if len(grp) >= 3 else 0)))
            if len(grp) >= 4 and len(grp) < 6:
                n_te = 3
            test.extend(grp[:n_te])
            train.extend(grp[n_te:])
        train.extend(by["others"])
        test.extend(by["others"])
    return train, test


def extract_file_windows(
    path: Path,
    window_ms: float,
    stride_ms: float,
    time_frac: Optional[Tuple[float, float]] = None,
    protocol: str = "old",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    imu = load_imu_abs(path)
    empty = (np.zeros((0, 1, 6), np.float32), np.zeros((0,), np.int64), [])
    if imu is None:
        return empty
    clips = labeled_clips(path, imu, protocol=protocol)
    xs, ys, ids = [], [], []
    for t0, t1, lab in clips:
        use_frac = time_frac if lab == "others" and label_from_name(path) == "others" else None
        if use_frac is not None:
            a, b = use_frac
            span = t1 - t0
            t0, t1 = t0 + a * span, t0 + b * span
        seq = resample_clip(imu, t0, t1)
        if seq is None:
            continue
        X = slice_windows(seq, window_ms, stride_ms)
        if len(X) == 0:
            continue
        xs.append(X)
        ys.append(np.full((len(X),), LABEL_TO_ID[lab], dtype=np.int64))
        ids.extend([f"{path.name}|{lab}" for _ in range(len(X))])
        print(f"[win] {path.name} proto={protocol} n={len(X)} label={lab} dur={(t1-t0)/1000:.2f}s")
    if not xs:
        return empty
    return np.concatenate(xs, 0), np.concatenate(ys, 0), ids


def build_xy(
    files: List[Path],
    window_ms: float,
    stride_ms: float,
    others_frac: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    recs = [ImuRec(p, "twophase" if is_twophase_name(p) else "old") for p in files]
    return build_xy_recs(recs, window_ms, stride_ms, others_frac)


def build_xy_recs(
    recs: List[ImuRec],
    window_ms: float,
    stride_ms: float,
    others_frac: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs, ys, rec_ids = [], [], []
    win = max(2, int(round(window_ms / 1000.0 * RATE_HZ)))
    for rec in recs:
        frac = others_frac if label_from_name(rec.path) == "others" else None
        X, y, ids = extract_file_windows(
            rec.path, window_ms, stride_ms, time_frac=frac, protocol=rec.protocol
        )
        if len(y) == 0:
            print(f"[skip windows] {rec.name} proto={rec.protocol}")
            continue
        xs.append(X)
        ys.append(y)
        rec_ids.extend(ids)
    if not xs:
        return np.zeros((0, win, 6), np.float32), np.zeros((0,), np.int64), []
    return np.concatenate(xs, 0), np.concatenate(ys, 0), rec_ids
