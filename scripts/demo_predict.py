"""Demo: run saved IMU/TP models on one recording and print majority vote."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grip_classify.predict import run_imu_file, run_tp_file


def _summarize(results, title: str):
    if not results:
        print(f"[{title}] no predictions")
        return
    counts = Counter(r["label"] for r in results)
    print(f"[{title}] windows={len(results)} majority={counts.most_common(1)[0]}")
    for lab, n in counts.most_common():
        print(f"  {lab}: {n} ({100.0 * n / len(results):.1f}%)")
    # show a few sample windows with proba
    for r in results[:: max(1, len(results) // 5)][:5]:
        proba = r.get("proba")
        if proba:
            top = sorted(proba.items(), key=lambda kv: -kv[1])[:2]
            top_s = ", ".join(f"{k}={v:.2f}" for k, v in top)
            print(f"  t={r['t_end_ms']:.0f} -> {r['label']} ({top_s})")
        else:
            print(f"  t={r['t_end_ms']:.0f} -> {r['label']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="右", help="Wzk subfolder label to demo")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data" / "wzk",
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=ROOT / "artifacts",
    )
    parser.add_argument("--modality", choices=["imu", "tp", "both"], default="both")
    args = parser.parse_args()

    folder = args.data_root / args.label
    if args.modality in ("imu", "both"):
        _summarize(
            run_imu_file(args.artifacts / "imu_model.joblib", folder / "imu_continuous.csv"),
            f"imu/{args.label}",
        )
    if args.modality in ("tp", "both"):
        _summarize(
            run_tp_file(
                args.artifacts / "tp_model.joblib", folder / "diffdata_continuous.jsonl"
            ),
            f"tp/{args.label}",
        )


if __name__ == "__main__":
    main()
