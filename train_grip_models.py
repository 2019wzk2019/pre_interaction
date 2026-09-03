"""CLI: train IMU and/or TP grip-state models on Wzk recordings."""

from __future__ import annotations

import argparse
from pathlib import Path

from grip_classify.data import load_all_recordings
from grip_classify.train import train_imu_model, train_tp_model


def main():
    parser = argparse.ArgumentParser(description="Train grip/reach state classifiers")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent / "Wzk",
        help="Folder containing 左to右 / 左 / 右to左 / 右 (左+右 -> others)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument(
        "--modality",
        choices=["imu", "tp", "both"],
        default="both",
    )
    parser.add_argument("--window-ms", type=float, default=150.0)
    args = parser.parse_args()

    recordings = load_all_recordings(args.data_root)
    if args.modality in ("imu", "both"):
        train_imu_model(
            args.data_root, args.out_dir, window_ms=args.window_ms, recordings=recordings
        )
    if args.modality in ("tp", "both"):
        train_tp_model(
            args.data_root, args.out_dir, window_ms=args.window_ms, recordings=recordings
        )


if __name__ == "__main__":
    main()
