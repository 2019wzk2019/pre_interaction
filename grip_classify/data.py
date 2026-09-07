"""Load IMU CSV and TP diffdata JSONL; reconstruct TP absolute levels via cumsum."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import (
    FOLDER_TO_LABEL,
    HEAD_EXCLUDE_FOLDERS,
    HEAD_EXCLUDE_MS,
    HOLDOUT_DIRNAME,
    LABEL_TO_ID,
    PRE_TAP_EXCLUDE_MS,
    SOURCE_FOLDERS,
    TAIL_EXCLUDE_MS,
    TP_GRID_COLS,
    TP_GRID_ROWS,
    TRAIN_DIRNAME,
)

_NUM_RE = re.compile(r"-?\d+")
_IMU_SESSION_RE = re.compile(r"^imu_continuous(?: \((\d+)\))?\.csv$", re.IGNORECASE)


def folder_class_name(folder_name: str) -> str:
    """Map on-disk folder to class. 'c左to右' in 测试集 -> 左to右."""
    if folder_name in FOLDER_TO_LABEL:
        return FOLDER_TO_LABEL[folder_name]
    if folder_name.startswith("c") and folder_name[1:] in FOLDER_TO_LABEL:
        return FOLDER_TO_LABEL[folder_name[1:]]
    raise KeyError(f"No class mapping for folder {folder_name!r}")


def list_label_dirs(data_root: Path) -> Dict[str, Path]:
    root = Path(data_root)
    train_root = root / TRAIN_DIRNAME if (root / TRAIN_DIRNAME).is_dir() else root
    found = discover_recording_dirs(train_root, skip_names={HOLDOUT_DIRNAME})
    missing = [n for n in SOURCE_FOLDERS if n not in found]
    if missing:
        print(f"[warn] expected train folders not found: {missing}")
    if not found:
        raise FileNotFoundError(f"No train recordings under {train_root}")
    print(f"[train dirs] {train_root}: {list(found.keys())}")
    return found


def list_holdout_dirs(test_root: Path) -> Dict[str, Path]:
    root = Path(test_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing holdout folder: {root}")
    found = discover_recording_dirs(root)
    if not found:
        raise FileNotFoundError(f"No recordings under {root}")
    print(f"[holdout dirs] {root}: {list(found.keys())}")
    return found


def discover_recording_dirs(
    root: Path,
    skip_names: Optional[set] = None,
    max_depth: int = 4,
) -> Dict[str, Path]:
    """Find folders that contain imu_continuous*.csv + matching TP jsonl."""
    skip_names = skip_names or set()
    found: Dict[str, Path] = {}

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth or not path.is_dir():
            return
        for child in sorted(path.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or child.name in skip_names:
                continue
            if list_recording_sessions(child):
                if child.name in found:
                    raise ValueError(f"duplicate recording folder name {child.name}")
                found[child.name] = child
            else:
                walk(child, depth + 1)

    walk(Path(root), 0)
    return found


def list_recording_sessions(folder: Path) -> List[Tuple[str, Path, Path, Path]]:
    """
    Pair IMU / TP / metadata files in a label folder.

    Supports a single take (`imu_continuous.csv`) or Windows copy suffixes
    (`imu_continuous (1).csv` ... `(n).csv`).
    """
    folder = Path(folder)
    sessions = []
    for imu_path in sorted(folder.iterdir()):
        if not imu_path.is_file():
            continue
        m = _IMU_SESSION_RE.match(imu_path.name)
        if not m:
            continue
        suffix = f" ({m.group(1)})" if m.group(1) else ""
        sid = m.group(1) or "0"
        tp_path = folder / f"diffdata_continuous{suffix}.jsonl"
        meta_path = folder / f"collection_run_metadata{suffix}.json"
        if not tp_path.is_file():
            print(f"[skip] {imu_path.name}: missing {tp_path.name}")
            continue
        if imu_path.stat().st_size < 64 or tp_path.stat().st_size < 64:
            print(f"[skip] empty session {imu_path.name}")
            continue
        sessions.append((sid, imu_path, tp_path, meta_path))
    return sessions


def load_imu(csv_path: Path) -> Dict[str, np.ndarray]:
    """Return dict with keys acc/gyro -> (N, 4) arrays [t_ms, x, y, z], sorted by time."""
    buckets: Dict[str, List[List[float]]] = {"acc": [], "gyro": []}
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sensor = (row.get("sensor") or "").strip()
            if sensor not in buckets:
                continue
            buckets[sensor].append(
                [
                    float(row["timestamp_ms"]),
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                ]
            )
    out = {}
    for sensor, rows in buckets.items():
        if not rows:
            raise ValueError(f"No {sensor} rows in {csv_path}")
        arr = np.asarray(rows, dtype=np.float64)
        order = np.argsort(arr[:, 0], kind="mergesort")
        out[sensor] = arr[order]
    return out


def _parse_tp_raw(raw: str) -> Tuple[np.ndarray, np.ndarray]:
    if "LineData=" in raw:
        grid_part, line_part = raw.split("LineData=", 1)
    else:
        grid_part, line_part = raw, ""
    grid_nums = [int(x) for x in _NUM_RE.findall(grid_part)]
    expected = TP_GRID_ROWS * TP_GRID_COLS
    if len(grid_nums) != expected:
        raise ValueError(f"Expected {expected} grid cells, got {len(grid_nums)}")
    grid = np.asarray(grid_nums, dtype=np.float64).reshape(TP_GRID_ROWS, TP_GRID_COLS)
    linedata = np.asarray([int(x) for x in _NUM_RE.findall(line_part)], dtype=np.float64)
    return grid, linedata


def load_tp_diff(jsonl_path: Path) -> Dict[str, np.ndarray]:
    """Load first-order TP diffs. Returns timestamps, grids (T,H,W), linedata (T,L)."""
    timestamps: List[float] = []
    grids: List[np.ndarray] = []
    lines: List[np.ndarray] = []
    with open(jsonl_path, encoding="utf-8") as f:
        first = f.readline()
        meta = json.loads(first)
        if meta.get("schema") != "preinteraction.diffdata.v1":
            # still try to parse as frames if not meta
            f.seek(0)
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "raw" not in obj:
                continue
            grid, linedata = _parse_tp_raw(obj["raw"])
            timestamps.append(float(obj["timestampMs"]))
            grids.append(grid)
            lines.append(linedata)

    if not timestamps:
        raise ValueError(f"No TP frames in {jsonl_path}")

    line_lens = {arr.shape[0] for arr in lines}
    if len(line_lens) != 1:
        # pad / truncate to mode length
        mode_len = max(line_lens, key=lambda n: sum(1 for a in lines if a.shape[0] == n))
        fixed = []
        for arr in lines:
            if arr.shape[0] == mode_len:
                fixed.append(arr)
            elif arr.shape[0] > mode_len:
                fixed.append(arr[:mode_len])
            else:
                pad = np.zeros(mode_len, dtype=np.float64)
                pad[: arr.shape[0]] = arr
                fixed.append(pad)
        lines = fixed

    t = np.asarray(timestamps, dtype=np.float64)
    order = np.argsort(t, kind="mergesort")
    return {
        "t_ms": t[order],
        "grid_diff": np.stack([grids[i] for i in order], axis=0),
        "line_diff": np.stack([lines[i] for i in order], axis=0),
        "poll_interval_ms": float(meta.get("pollIntervalMs", 12)) if isinstance(meta, dict) else 12.0,
    }


def reconstruct_tp_absolute(
    grid_diff: np.ndarray,
    line_diff: np.ndarray,
    start_grid: Optional[np.ndarray] = None,
    start_line: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert first-order diffs to relative absolute levels via cumulative sum.

    Absolute value at frame t: base + sum(diff[0..t]).
    Default base is zeros so levels are relative to the recording start
    (preserves spatial pattern / temporal state without needing a true DC baseline).
    """
    if start_grid is None:
        start_grid = np.zeros(grid_diff.shape[1:], dtype=np.float64)
    if start_line is None:
        start_line = np.zeros(line_diff.shape[1], dtype=np.float64)
    grid_abs = np.cumsum(grid_diff, axis=0) + start_grid
    line_abs = np.cumsum(line_diff, axis=0) + start_line
    return grid_abs, line_abs


def load_tap_times_ms(meta_path: Path) -> List[float]:
    """Finger-down / free_tap timestamps from collection_run_metadata.json."""
    if not meta_path.is_file():
        return []
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    times = []
    for ev in meta.get("interactionEvents") or []:
        ts = ev.get("timestampMs")
        if ts is not None:
            times.append(float(ts))
    for ev in meta.get("touchEvents") or []:
        if ev.get("phase") == "down" and ev.get("timestampMs") is not None:
            times.append(float(ev["timestampMs"]))
    if not times:
        return []
    arr = np.unique(np.round(np.asarray(times, dtype=np.float64), 3))
    return arr.tolist()


def build_exclude_intervals(
    folder_name: str,
    t_start_ms: float,
    t_end_ms: float,
    tap_times_ms: List[float],
    class_name: Optional[str] = None,
) -> List[Tuple[float, float]]:
    """
    Windows overlapping these [start, end] intervals are dropped from the dataset.

    - Reach classes (左to右 / 右to左, including 测试集 c* folders): drop the first HEAD_EXCLUDE_MS.
    - Any recording: drop PRE_TAP_EXCLUDE_MS before each TP tap (finger down).
    - Any recording: drop the last TAIL_EXCLUDE_MS (end of take is often messy).
    """
    intervals: List[Tuple[float, float]] = []
    reach = class_name in ("左to右", "右to左") or folder_name in HEAD_EXCLUDE_FOLDERS
    if reach:
        intervals.append((float(t_start_ms), float(t_start_ms) + HEAD_EXCLUDE_MS))
    for tap in tap_times_ms:
        intervals.append((float(tap) - PRE_TAP_EXCLUDE_MS, float(tap)))
    tail_start = float(t_end_ms) - TAIL_EXCLUDE_MS
    intervals.append((tail_start, float(t_end_ms) + 1.0))
    intervals.sort()
    merged: List[Tuple[float, float]] = []
    for a, b in intervals:
        if b <= a:
            continue
        if not merged or a > merged[-1][1]:
            merged.append((a, b))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
    return merged


def window_overlaps_excludes(
    t_start: float,
    t_end: float,
    intervals: List[Tuple[float, float]],
) -> bool:
    for a, b in intervals:
        if t_start < b and t_end > a:
            return True
    return False


def _load_one_recording(
    folder_name: str,
    imu_path: Path,
    tp_path: Path,
    meta_path: Path,
    session_id: str = "0",
) -> dict:
    imu = load_imu(imu_path)
    tp_diff = load_tp_diff(tp_path)
    grid_abs, line_abs = reconstruct_tp_absolute(
        tp_diff["grid_diff"], tp_diff["line_diff"]
    )
    tap_times = load_tap_times_ms(meta_path) if meta_path.is_file() else []
    t_start = float(min(imu["acc"][0, 0], imu["gyro"][0, 0], tp_diff["t_ms"][0]))
    t_end = float(max(imu["acc"][-1, 0], imu["gyro"][-1, 0], tp_diff["t_ms"][-1]))
    class_name = folder_class_name(folder_name)
    excludes = build_exclude_intervals(
        folder_name, t_start, t_end, tap_times, class_name=class_name
    )
    rec_name = folder_name if session_id == "0" else f"{folder_name}#{session_id}"
    rec = {
        "folder": rec_name,
        "label": class_name,
        "label_id": LABEL_TO_ID[class_name],
        "imu": imu,
        "tap_times_ms": tap_times,
        "exclude_intervals": excludes,
        "t_start_ms": t_start,
        "t_end_ms": t_end,
        "tp": {
            "t_ms": tp_diff["t_ms"],
            "grid_diff": tp_diff["grid_diff"],
            "line_diff": tp_diff["line_diff"],
            "grid_abs": grid_abs,
            "line_abs": line_abs,
            "poll_interval_ms": tp_diff["poll_interval_ms"],
        },
    }
    dur_s = (t_end - t_start) / 1000.0
    print(
        f"[load] {rec_name} -> {class_name}: {dur_s:.1f}s acc={len(imu['acc'])} "
        f"gyro={len(imu['gyro'])} tp={len(tp_diff['t_ms'])} frames taps={len(tap_times)} "
        f"exclude={len(excludes)} (incl. last {TAIL_EXCLUDE_MS}ms)"
    )
    return rec


def _load_folder_recordings(folder_name: str, folder: Path) -> Dict[str, dict]:
    sessions = list_recording_sessions(folder)
    if not sessions:
        raise FileNotFoundError(f"No IMU+TP sessions in {folder}")
    out = {}
    for sid, imu_path, tp_path, meta_path in sessions:
        try:
            rec = _load_one_recording(
                folder_name, imu_path, tp_path, meta_path, session_id=sid
            )
        except Exception as e:
            print(f"[skip] {imu_path}: {e}")
            continue
        out[rec["folder"]] = rec
    if not out:
        raise FileNotFoundError(f"No valid IMU+TP sessions in {folder}")
    return out


def load_all_recordings(data_root: Path) -> Dict[str, dict]:
    """Load IMU + reconstructed TP for every training label folder (not 测试集)."""
    dirs = list_label_dirs(data_root)
    out: Dict[str, dict] = {}
    for name, folder in dirs.items():
        out.update(_load_folder_recordings(name, folder))
    return out


def load_holdout_recordings(test_root: Optional[Path] = None, data_root: Optional[Path] = None) -> Dict[str, dict]:
    """Load Wzk/测试集 (or explicit test_root) as a held-out evaluation set."""
    if test_root is None:
        if data_root is None:
            raise ValueError("need test_root or data_root")
        test_root = Path(data_root) / HOLDOUT_DIRNAME
    dirs = list_holdout_dirs(test_root)
    out: Dict[str, dict] = {}
    for name, folder in dirs.items():
        out.update(_load_folder_recordings(name, folder))
    return out
