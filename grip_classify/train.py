"""Train RandomForest classifiers for IMU and TP grip-state prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import ID_TO_LABEL, LABELS, WINDOW_MS
from .data import load_all_recordings
from .features import make_imu_dataset, make_tp_dataset


def _build_clf(n_estimators: int = 300, random_state: int = 42) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=24,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _eval(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
    return {
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "accuracy": float(report["accuracy"]),
    }


def train_imu_model(
    data_root: Path,
    out_dir: Path,
    window_ms: float = WINDOW_MS,
    recordings: Optional[Dict] = None,
) -> Tuple[Pipeline, Dict]:
    if recordings is None:
        recordings = load_all_recordings(data_root)
    Xtr, ytr, Xte, yte = make_imu_dataset(recordings, window_ms=window_ms)
    print(f"[imu] train={len(ytr)} test={len(yte)} feat_dim={Xtr.shape[1]}")
    clf = _build_clf()
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = _eval(yte, pred)
    print(classification_report(yte, pred, target_names=LABELS, digits=4, zero_division=0))
    print("confusion_matrix:\n", confusion_matrix(yte, pred, labels=list(range(len(LABELS)))))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "imu_model.joblib"
    joblib.dump(
        {
            "pipeline": clf,
            "labels": LABELS,
            "id_to_label": ID_TO_LABEL,
            "window_ms": window_ms,
            "modality": "imu",
            "feature_dim": int(Xtr.shape[1]),
        },
        model_path,
    )
    with open(out_dir / "imu_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[imu] saved {model_path} acc={metrics['accuracy']:.4f}")
    return clf, metrics


def train_tp_model(
    data_root: Path,
    out_dir: Path,
    window_ms: float = WINDOW_MS,
    recordings: Optional[Dict] = None,
) -> Tuple[Pipeline, Dict]:
    if recordings is None:
        recordings = load_all_recordings(data_root)
    Xtr, ytr, Xte, yte = make_tp_dataset(recordings, window_ms=window_ms)
    print(f"[tp] train={len(ytr)} test={len(yte)} feat_dim={Xtr.shape[1]}")
    clf = _build_clf(n_estimators=400)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = _eval(yte, pred)
    print(classification_report(yte, pred, target_names=LABELS, digits=4, zero_division=0))
    print("confusion_matrix:\n", confusion_matrix(yte, pred, labels=list(range(len(LABELS)))))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "tp_model.joblib"
    joblib.dump(
        {
            "pipeline": clf,
            "labels": LABELS,
            "id_to_label": ID_TO_LABEL,
            "window_ms": window_ms,
            "modality": "tp",
            "feature_dim": int(Xtr.shape[1]),
            "uses_cumsum_absolute": True,
            "window_demean": True,
            "tp_edge_only": True,
        },
        model_path,
    )
    with open(out_dir / "tp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[tp] saved {model_path} acc={metrics['accuracy']:.4f}")
    return clf, metrics
