"""CLI: train IMU and/or TP grip-state models on data/wzk recordings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grip_classify.data import load_all_recordings, load_holdout_recordings
from grip_classify.train import train_imu_model, train_tp_model


def main():
    parser = argparse.ArgumentParser(description="Train grip/reach state classifiers")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "wzk",
        help="Root with 训练集/ and 测试集/ (or flat label folders)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts",
    )
    parser.add_argument(
        "--modality",
        choices=["imu", "tp", "both"],
        default="both",
    )
    parser.add_argument("--window-ms", type=float, default=150.0)
    parser.add_argument(
        "--holdout-dir",
        type=Path,
        default=None,
        help="Held-out test recordings (default: <data-root>/测试集 if present)",
    )
    parser.add_argument(
        "--no-holdout",
        action="store_true",
        help="Ignore 测试集 and use per-recording 80/20 temporal split",
    )
    args = parser.parse_args()

    recordings = load_all_recordings(args.data_root)
    holdout = None
    holdout_dir = args.holdout_dir
    if holdout_dir is None:
        holdout_dir = args.data_root / "测试集"
    if not args.no_holdout and holdout_dir.is_dir():
        print(f"[holdout] loading {holdout_dir}")
        holdout = load_holdout_recordings(holdout_dir)
    elif not args.no_holdout:
        print(f"[holdout] not found at {holdout_dir}, falling back to 80/20 split")

    if args.modality in ("imu", "both"):
        train_imu_model(
            args.data_root,
            args.out_dir,
            window_ms=args.window_ms,
            recordings=recordings,
            holdout_recordings=holdout,
        )
    if args.modality in ("tp", "both"):
        for robust, suffix in ((True, "_p298"), (False, "_raw")):
            train_tp_model(
                args.data_root,
                args.out_dir,
                window_ms=args.window_ms,
                recordings=recordings,
                holdout_recordings=holdout,
                robust_p2_p98=robust,
                name_suffix=suffix,
            )


if __name__ == "__main__":
    main()
