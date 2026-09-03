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
    LABEL_TO_ID,
    PRE_TAP_EXCLUDE_MS,
    SOURCE_FOLDERS,
    TP_GRID_COLS,
    TP_GRID_ROWS,
)

_NUM_RE = re.compile(r"-?\d+")


def list_label_dirs(data_root: Path) -> Dict[str, Path]:
    root = Path(data_root)
    found = {}
    for name in SOURCE_FOLDERS:
        p = root / name
        if not p.is_dir():
            raise FileNotFoundError(f"Missing source folder: {p}")
        found[name] = p
    return found


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
    tap_times_ms: List[float],
) -> List[Tuple[float, float]]:
    """
    Windows overlapping these [start, end] intervals are dropped from the dataset.

    - 左to右 / 右to左: drop the first HEAD_EXCLUDE_MS of the recording.
    - Any recording: drop PRE_TAP_EXCLUDE_MS before each TP tap (finger down).
    """
    intervals: List[Tuple[float, float]] = []
    if folder_name in HEAD_EXCLUDE_FOLDERS:
        intervals.append((float(t_start_ms), float(t_start_ms) + HEAD_EXCLUDE_MS))
    for tap in tap_times_ms:
        intervals.append((float(tap) - PRE_TAP_EXCLUDE_MS, float(tap)))
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


def load_all_recordings(data_root: Path) -> Dict[str, dict]:
    """Load IMU + reconstructed TP for every label folder."""
    dirs = list_label_dirs(data_root)
    out = {}
    for folder_name, folder in dirs.items():
        imu = load_imu(folder / "imu_continuous.csv")
        tp_diff = load_tp_diff(folder / "diffdata_continuous.jsonl")
        grid_abs, line_abs = reconstruct_tp_absolute(
            tp_diff["grid_diff"], tp_diff["line_diff"]
        )
        tap_times = load_tap_times_ms(folder / "collection_run_metadata.json")
        t_start = float(min(imu["acc"][0, 0], imu["gyro"][0, 0], tp_diff["t_ms"][0]))
        excludes = build_exclude_intervals(folder_name, t_start, tap_times)
        class_name = FOLDER_TO_LABEL[folder_name]
        out[folder_name] = {
            "folder": folder_name,
            "label": class_name,
            "label_id": LABEL_TO_ID[class_name],
            "imu": imu,
            "tap_times_ms": tap_times,
            "exclude_intervals": excludes,
            "tp": {
                "t_ms": tp_diff["t_ms"],
                "grid_diff": tp_diff["grid_diff"],
                "line_diff": tp_diff["line_diff"],
                "grid_abs": grid_abs,
                "line_abs": line_abs,
                "poll_interval_ms": tp_diff["poll_interval_ms"],
            },
        }
        print(
            f"[load] {folder_name} -> {class_name}: acc={len(imu['acc'])} gyro={len(imu['gyro'])} "
            f"tp={len(tp_diff['t_ms'])} frames taps={len(tap_times)} exclude={len(excludes)}"
        )
    return out
