"""Export latest artifacts/*.joblib into HarmonyOS rawfile JSON."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from export_harmony_models import _export_rf
from grip_classify import LABELS, WINDOW_MS


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = (
        root
        / "harmony_app"
        / "GripReachPredict"
        / "entry"
        / "src"
        / "main"
        / "resources"
        / "rawfile"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    extras = {
        "imu": None,
        "tp": {
            "uses_cumsum_absolute": True,
            "window_demean": True,
            "tp_grid_rows": 34,
            "tp_grid_cols": 21,
        },
    }
    for name, extra in extras.items():
        bundle = joblib.load(root / "artifacts" / f"{name}_model.joblib")
        pipe = bundle["pipeline"]
        rf = pipe.named_steps["rf"]
        scaler = pipe.named_steps["scaler"]
        labels = bundle.get("labels", LABELS)
        window_ms = bundle.get("window_ms", WINDOW_MS)
        print(
            f"{name}: trees={len(rf.estimators_)} "
            f"feat={scaler.mean_.shape[0]} labels={labels}"
        )
        obj = _export_rf(pipe, labels, name, window_ms, extra=extra)
        path = out_dir / f"{name}_model.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))
        print(f"  wrote {path} size={path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
