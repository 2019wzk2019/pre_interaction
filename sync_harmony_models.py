"""Export joblib models to HarmonyOS rawfile JSON (and a shareable delivery folder)."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import joblib

from export_harmony_models import _export_rf
from grip_classify import LABELS, WINDOW_MS


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
    print(f"  wrote {path} size={path.stat().st_size / 1024:.1f} KB")


def export_one(bundle_path: Path, extra: Optional[dict], modality: str) -> dict:
    bundle = joblib.load(bundle_path)
    pipe = bundle["pipeline"]
    rf = pipe.named_steps["rf"]
    scaler = pipe.named_steps["scaler"]
    labels = bundle.get("labels", LABELS)
    window_ms = bundle.get("window_ms", WINDOW_MS)
    extra_out = dict(extra) if extra else {}
    for k in (
        "uses_cumsum_absolute",
        "window_demean",
        "tp_edge_only",
        "robust_p2_p98",
    ):
        if k in bundle:
            extra_out[k] = bundle[k]
    print(
        f"{modality}: src={bundle_path.name} trees={len(rf.estimators_)} "
        f"feat={scaler.mean_.shape[0]} robust_p2_p98={extra_out.get('robust_p2_p98')}"
    )
    return _export_rf(pipe, labels, modality, window_ms, extra=extra_out or None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-joblib", type=Path, default=None)
    parser.add_argument("--imu-joblib", type=Path, default=None)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    art = root / "artifacts"
    hap_raw = (
        root
        / "harmony_app"
        / "GripReachPredict"
        / "entry"
        / "src"
        / "main"
        / "resources"
        / "rawfile"
    )
    delivery = art / "hap_delivery"

    tp_joblib = args.tp_joblib
    if tp_joblib is None:
        raw = art / "tp_model_raw.joblib"
        tp_joblib = raw if raw.is_file() else art / "tp_model.joblib"
    imu_joblib = args.imu_joblib or (art / "imu_model.joblib")

    tp_extra = {
        "uses_cumsum_absolute": True,
        "window_demean": True,
        "tp_edge_only": True,
        "robust_p2_p98": False,
        "tp_grid_rows": 34,
        "tp_grid_cols": 21,
        "tp_edge_layers": 3,
    }

    jobs = [("tp", tp_joblib, tp_extra)]
    if imu_joblib.is_file():
        jobs.insert(0, ("imu", imu_joblib, None))

    for name, joblib_path, extra in jobs:
        obj = export_one(joblib_path, extra, name)
        if name == "tp" and obj["feature_dim"] != 1021:
            print(f"  WARNING: expected feat_dim≈1021, got {obj['feature_dim']}")
        if name == "tp" and obj.get("robust_p2_p98") is not False:
            print("  WARNING: expected robust_p2_p98=false for this delivery")
        _write_json(obj, delivery / f"{name}_model.json")
        _write_json(obj, hap_raw / f"{name}_model.json")

    adapt = root / "docs" / "HAP_TP_ADAPT.md"
    if adapt.is_file():
        shutil.copy2(adapt, delivery / "HAP_TP_ADAPT.md")
        print(f"  copied {adapt.name} -> {delivery / 'HAP_TP_ADAPT.md'}")
    print(f"delivery folder: {delivery}")


if __name__ == "__main__":
    main()
