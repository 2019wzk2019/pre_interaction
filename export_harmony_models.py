"""Train mobile-sized RFs and export scaler+trees JSON for HarmonyOS rawfile."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from grip_classify import ID_TO_LABEL, LABELS, WINDOW_MS
from grip_classify.data import load_all_recordings
from grip_classify.features import make_imu_dataset, make_tp_dataset


def _export_rf(pipeline: Pipeline, labels, modality: str, window_ms: float, extra=None) -> dict:
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    rf: RandomForestClassifier = pipeline.named_steps["rf"]
    trees = []
    for est in rf.estimators_:
        t = est.tree_
        values = t.value[:, 0, :].astype(np.float64)
        row_sum = values.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        probs = values / row_sum
        trees.append(
            {
                "children_left": t.children_left.astype(np.int32).tolist(),
                "children_right": t.children_right.astype(np.int32).tolist(),
                "feature": t.feature.astype(np.int32).tolist(),
                "threshold": t.threshold.astype(np.float64).tolist(),
                "value": probs.tolist(),
            }
        )
    out = {
        "modality": modality,
        "labels": list(labels),
        "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()},
        "window_ms": float(window_ms),
        "feature_dim": int(scaler.mean_.shape[0]),
        "n_estimators": len(trees),
        "scaler_mean": scaler.mean_.astype(np.float64).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
        "trees": trees,
    }
    if extra:
        out.update(extra)
    return out


def _train(Xtr, ytr, Xte, yte, n_estimators: int, max_depth: int) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    print(classification_report(yte, pred, target_names=LABELS, digits=4, zero_division=0))
    print("cm:\n", confusion_matrix(yte, pred, labels=list(range(len(LABELS)))))
    print("acc=", accuracy_score(yte, pred))
    return pipe


def main():
    root = Path(__file__).resolve().parent
    data_root = root / "Wzk"
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
    recordings = load_all_recordings(data_root)

    print("=== IMU mobile RF ===")
    Xtr, ytr, Xte, yte = make_imu_dataset(recordings)
    imu_pipe = _train(Xtr, ytr, Xte, yte, n_estimators=80, max_depth=16)
    imu_path = out_dir / "imu_model.json"
    with open(imu_path, "w", encoding="utf-8") as f:
        json.dump(_export_rf(imu_pipe, LABELS, "imu", WINDOW_MS), f, ensure_ascii=False, separators=(",", ":"))
    print(f"wrote {imu_path} size={imu_path.stat().st_size / 1024:.1f} KB")

    print("=== TP mobile RF ===")
    Xtr, ytr, Xte, yte = make_tp_dataset(recordings)
    tp_pipe = _train(Xtr, ytr, Xte, yte, n_estimators=60, max_depth=14)
    tp_path = out_dir / "tp_model.json"
    with open(tp_path, "w", encoding="utf-8") as f:
        json.dump(
            _export_rf(
                tp_pipe,
                LABELS,
                "tp",
                WINDOW_MS,
                extra={
                    "uses_cumsum_absolute": True,
                    "window_demean": True,
                    "tp_grid_rows": 34,
                    "tp_grid_cols": 21,
                },
            ),
            f,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    print(f"wrote {tp_path} size={tp_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
